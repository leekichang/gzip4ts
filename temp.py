import gzip
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
from multiprocessing import Pool

import ops
import utils
import config as cfg
from DataManager import DataManager

class Knn(object):
    def __init__(self, trainset, args):
        self.trainset  = trainset
        self.dec_point = args.decimal
        self.K         = args.k
    
    def run_diff(self, data):
        x1, _ = data
        distance_from_x1 = []
        for x2, _ in self.trainset:
            distance_from_x1.append(np.sum(x1-x2))
        return distance_from_x1

    def run_cosine(self, data):
        x1, _ = data
        distance_from_x1 = []
        for x2, _ in self.trainset:
            distance_from_x1.append(np.sum(x1.reshape(-1)*x2.reshape(-1)))
        return distance_from_x1

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
    
    def run(self, testset, compress='quant_fp'):
        pred = []
        with Pool(processes=None) as pool:
            distance_lists = list(tqdm(pool.imap(getattr(self, f'run_{compress}'), testset), total=len(testset)))

        for distance_from_x1 in distance_lists:
            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_class = self.trainset.Y[sorted_idx[:self.K]].tolist()
            predict_class = max(set(top_k_class), key=top_k_class.count)
            pred.append(predict_class)
        return testset.Y.tolist(), pred

def load_dataset(path="../dataset", dataset='mitbih_arr'):
    X = np.load(f'{path}/{dataset}/x_train.npy')
    Y = np.load(f'{path}/{dataset}/y_train.npy')
    trainset = DataManager(X, Y)
    X = np.load(f'{path}/{dataset}/x_test.npy')
    Y = np.load(f'{path}/{dataset}/y_test.npy')
    testset = DataManager(X, Y)
    return trainset, testset

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
    print(args.dataset, max(trainset.Y))
    trainset = choose_trainset(trainset, args)

    # testset[:,0] = ops.moving_average(testset[:, 0], 3)
    # trainset[:,0] = ops.moving_average(trainset[:, 0], 3)
    # testset.X  = ops.calculate_derivative(testset.X)
    # trainset.X = ops.calculate_derivative(trainset.X)
    
    knn = Knn(trainset=trainset, args=args)
    y_true, y_pred = knn.run(testset, compress=f'{args.dtype}')
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    bacc = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    print()
    print(f'         ACC: { acc*100:>5.2f}%' , flush=True)
    print(f'Balanced ACC: {bacc*100:>5.2f}%', flush=True)
    print(conf_mat, flush=True)
    # utils.save_result(args, y_true, y_pred)
    # utils.load_result(args)