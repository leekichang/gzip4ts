import numpy as np
from utils import ExpLogger, arguments

def test_ExpLogger():
    logger = ExpLogger(
        "../results/journal",
        log_accuracy=True,
        log_memory=True,
        log_time=True
    )
    
    import time
    
    logger.start_measure_memory()
    logger.start_measure_time("test")
    time.sleep(2)
    logger.end_measure_time("test")
    logger.start_measure_time("test")
    time.sleep(2)
    logger.end_measure_time("test")
    logger.end_measure_memory()

def test_rawdataset(args):
    from data.utils import load_raw_dataset
    trainset, testset = load_raw_dataset(
        dataset=args.dataset,
    )
    
    print(trainset.X.shape, trainset.Y.shape)
    print(testset.X.shape, testset.Y.shape)

def run_gzipClassifier(args):
    
    np.random.seed(args.seed)
    
    from classifiers.gzip_classifier import GzipClassifier
    from sklearn import metrics
    
    logger = ExpLogger(
        "../results/journal",
        exp_name=f"{args.exp_name}",
        tag=f"{args.dataset}_{args.n_shots}_{args.decimal}_{args.k}_{args.method}_{args.benchmark}_{args.seed}",
        default_overwrite=False,
        log_start_time=False,
        log_accuracy=True,
        log_memory=True,
        log_time=True
    )
    
    if logger.terminate == True:
        return
    
    logger.start_measure_memory()
    logger.start_measure_time("all")
    classfier = GzipClassifier(args)
    y_true, y_pred = classfier.run()
    logger.end_measure_time("all")
    del classfier
    logger.end_measure_memory()
    
    # acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    bacc = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    # conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    logger.write_accuracy("bacc", bacc)
    logger.write_accuracy("conf_mat", metrics.confusion_matrix(y_true=y_true, y_pred=y_pred))
    # print(metrics.classification_report(y_true, y_pred))
    print(f"bacc: {bacc}")

def exp1_rungzip_variousparam_0724():
    """
    Accuracy is precise
    others are not precise
    """
    
    from tqdm import tqdm
    import argparse
    import itertools
    
    exp_name = "exp1_rungzip_variousparam_0724"
    datasets = ["mitbih_arr", "pamap2", "mitbih_auth"]
    n_shots = [1, 2, 3, 4, 8, 16, 32] # [1, 2, 3, 4, 8, 16, 32, 64]
    ks = [1, 3, 5, 7, 9, 21] # [1, 3, 5, 7, 9, 21]
    decimals = [1, 2, 3, 4, 5] # [ 1, 2, 3, 4, 5, 6, 7, 8 ,16]
    seeds = range(3)
    methods = ["default", "all"]
    
    combinations = list(itertools.product(datasets, n_shots, methods, decimals, ks, seeds))
    
    for dataset, n_shot, method, decimal, k, seed in tqdm(combinations):
        if method == "default":
            if decimal != 1:
                continue
            else:
                decimal = None
                
        args = argparse.Namespace(
            exp_name=exp_name,
            dataset=dataset,
            seed=seed,
            n_shots=n_shot,
            decimal=decimal,
            k=k,
            method=method,
            benchmark=False
        )
        
        run_gzipClassifier(args)

if __name__ == '__main__':
    # args = arguments.parse_args()
    
    # import pprint
    # pprint.pprint(args)
    
    # np.random.seed(args.seed)
    
    # test_ExpLogger()
    exp1_rungzip_variousparam_0724()
    
    
    