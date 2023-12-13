import gzip
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
from multiprocessing import Pool
from scipy.stats import entropy
from scipy.special import kl_div
import config as cfg

import ops
import utils
from DataManager import DataManager

class Knn(object):
    def __init__(self, trainset, args):
        self.trainset  = trainset
        self.dec_point = args.decimal
        self.K         = args.k
    

    def run_str(self, data):
        x1, _ = data
        s1 = ' '.join(x1.astype(str))
        Cx1 = len(gzip.compress(s1.encode()))
        distance_from_x1 = []
        for x2, _ in self.trainset:
            s2 = ' '.join(x2.astype(str))
            Cx2 = len(gzip.compress(s2.encode()))
            x1x2 = ' '.join([s1, s2])
            Cx1x2 = len(gzip.compress(x1x2.encode()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        return distance_from_x1
        
    def run_quant_str(self, data):
        x1, _ = data
        s1 = ' '.join(np.round(x1, self.dec_point).astype(str))
        Cx1 = len(gzip.compress(s1.encode()))
        distance_from_x1 = []
        for x2, _ in self.trainset:
            s2 = ' '.join(np.round(x2, self.dec_point).astype(str))
            Cx2 = len(gzip.compress(s2.encode()))
            x1x2 = ' '.join([s1, s2])
            Cx1x2 = len(gzip.compress(x1x2.encode()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        return distance_from_x1
    
    def run_quant_fp(self, data):
        x1, _ = data
        x1 = np.round(x1, self.dec_point)
        Cx1 = len(gzip.compress(x1.tobytes()))
        distance_from_x1 = []
        for x2, _ in self.trainset:
            x2 = np.round(x2, self.dec_point)
            Cx2 = len(gzip.compress(x2.tobytes()))
            x1x2 = np.concatenate([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.tobytes()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        return distance_from_x1
    
    def run_fp(self, data):
        x1, _ = data
        Cx1 = len(gzip.compress(x1.tobytes()))
        distance_from_x1 = []
        for x2, _ in self.trainset:
            Cx2 = len(gzip.compress(x2.tobytes()))
            x1x2 = np.concatenate([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.tobytes()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        return distance_from_x1
    
    def run_fft_hist_entropy_ncd(self, data):
        x1, _ = data
        x1 = x1[0] # assume single channel
        Cx1 = entropy(x1)
        distance_from_x1 = []
        for x2, _ in self.trainset:
            x2 = x2[0]
            Cx2 = entropy(x2)
            x1x2 = x1 + x2
            Cx1x2 = entropy(x1x2)
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        return distance_from_x1
    
    def run_fft_hist_entropy_kldiv(self, data):
        x1, _ = data
        x1 = x1[0] # assume single channel
        # Cx1 = entropy(x1)
        distance_from_x1 = []
        for x2, _ in self.trainset:
            x2 = x2[0]
            # Cx2 = entropy(x2)
            # x1x2 = x1 + x2
            # Cx1x2 = entropy(x1x2)
            # ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            kl_divergence = sum(kl_div(x1, x2))
            distance_from_x1.append(kl_divergence)
        return distance_from_x1
    
    def run_fft_both_entropy_kldiv(self, data):
        x1, _ = data
        a1 = x1[0] # assume single channel
        p1 = x1[1]
        # Cx1 = entropy(x1)
        distance_from_x1 = []
        for x2, _ in self.trainset:
            a2 = x2[0]
            p2 = x2[1]
            
            kl_divergence = sum(kl_div(a1, a2)) - sum(kl_div(p1, p2))
            distance_from_x1.append(kl_divergence)
        return distance_from_x1
    
    def run(self, testset, compress='quant_fp'):
        pred = []
        with Pool(processes=None) as pool:
            distance_lists = list(tqdm(
                pool.imap(getattr(self, f'run_{compress}'), testset),
                total=len(testset)
            ))

        for distance_from_x1 in distance_lists:
            distance_from_x1 = np.array(distance_from_x1)
            sorted_idx = np.argsort(distance_from_x1)
            distance = distance_from_x1[sorted_idx[:self.K]]
            
            # # -- average (distance low, # near high)
            # top_k_class = self.trainset.Y[sorted_idx[:self.K]]
            
            # distance_per_class = {
            #     class_idx: np.average(distance[top_k_class == class_idx]) for class_idx in set(top_k_class)
            # }
            # predict_class = min(set(top_k_class), key=lambda class_idx: distance_per_class[class_idx])
            
            # -- hard voting
            top_k_class = self.trainset.Y[sorted_idx[:self.K]]
            unique_elements, counts = np.unique(top_k_class, return_counts=True)    

            most_frequent_indices = np.where((counts == counts.max()))[0]
            
            most_frequent_classes = unique_elements[most_frequent_indices]
            min_distances = [min(distance[top_k_class==most_frequent_class]) for most_frequent_class in most_frequent_classes]
            
            predict_class = most_frequent_classes[np.argmin(min_distances)]

            
            # # -- average ?
            # top_k_class = self.trainset.Y[sorted_idx[:self.K]]
            # distance = distance_from_x1[sorted_idx[:self.K]]
            
            # distance_per_class = {
            #     class_idx: np.average(distance[top_k_class == class_idx]) for class_idx in set(top_k_class)
            # }
            # predict_class = min(set(top_k_class), key=lambda class_idx: distance_per_class[class_idx])
            
            
            pred.append(predict_class)
        return testset.Y.tolist(), pred

# def load_dataset(path="../dataset"):
#     task = "arr"
    
#     X = np.load(f'{path}/train/{task}_X.npy')
#     Y = np.load(f'{path}/train/{task}_Y.npy')
#     trainset = DataManager(X, Y)
#     X = np.load(f'{path}/test/{task}_X.npy')
#     Y = np.load(f'{path}/test/{task}_Y.npy')
#     testset = DataManager(X, Y)
#     return trainset, testset

def load_dataset(path="../dataset", dataset='mitbih_arr'):
    X = np.load(f'{path}/{dataset}/x_train.npy')
    Y = np.load(f'{path}/{dataset}/y_train.npy')
    trainset = DataManager(X, Y)
    X = np.load(f'{path}/{dataset}/x_test.npy')
    Y = np.load(f'{path}/{dataset}/y_test.npy')
    testset = DataManager(X, Y)
    return trainset, testset

# # n_class => id: 48, gender: 2, arr: 6
# def choose_trainset(trainset, num_shots=3, num_class=6, seed=6206):
#     np.random.seed(seed)
#     X = []
#     Y = np.array([])
#     for i in range(num_class):
#         idx = np.random.choice(len(trainset.Y[trainset.Y==i]), num_shots, replace=False)
#         X.append(trainset.X[trainset.Y==i][idx])
#         Y = np.hstack([Y, np.ones(num_shots)*i])
#     X = np.concatenate(X, axis=0) # (num_class * num_shots, n_channel, timeseries_length)
#     trainset = DataManager(X, Y)
#     return trainset

def choose_trainset(trainset, args):
    np.random.seed(args.seed)
    X = []
    Y = np.array([])
    for i in range(cfg.N_CLASS[args.dataset]):
        idx = np.random.choice(len(trainset.Y[trainset.Y==i]), args.n_shots, replace=False)
        X.append(trainset.X[trainset.Y==i][idx])
        Y = np.hstack([Y, np.ones(args.n_shots)*i])
    X = np.concatenate(X, axis=0)
    trainset = DataManager(X, Y)
    return trainset

if __name__ == '__main__':
    args = utils.parse_args()
    print(f'exp_name:{args.exp_name}\ndecimal:{args.decimal}\nnum shot:{args.n_shots}\ncompress type:{args.dtype}')
    trainset, testset = load_dataset(dataset=args.dataset)
    trainset = choose_trainset(trainset, args)
    
    # testset.X  = ops.calculate_derivative(testset.X)
    # trainset.X = ops.calculate_derivative(trainset.X)
    
    trainset.X = ops.fft(trainset.X)
    testset.X  = ops.fft(testset.X)
    
    knn = Knn(trainset=trainset, args=args)
    y_true, y_pred = knn.run(testset, compress=f'{args.dtype}')
    b_acc = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    print()
    print(f'balanced:{b_acc*100:.2f}%', flush=True)
    print(f'non-balanced:{acc*100:.2f}%', flush=True)
    print(conf_mat, flush=True)
    # utils.save_result(args, y_true, y_pred)
    # utils.load_result(args)