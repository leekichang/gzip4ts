import argparse
import gc
import time

import numpy as np
from memory_profiler import profile
from sklearn import metrics
import torch
from classifiers.gzip_classifier import GzipClassifier
from classifiers.shallow_classifier import MLTrainer as ShallowClassifier
from classifiers.deep_classifier import SupervisedTrainer as DeepClassifier
from utils.arguments import str2bool, parse_args
from utils.logger import ExpLogger

# @profile
def run_deepClassifier(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger = ExpLogger(
        "../results/journal",
        exp_name=f"{args.exp_name}",
        tag=f"{args.dataset}_{args.n_shots}_{args.model}_{args.epochs}_{args.batch_size}_{args.benchmark}_{args.seed}",
        default_overwrite=args.overwrite,
        log_start_time=False,
        log_accuracy=not args.benchmark,
        log_memory=args.benchmark,
        log_time=False
    )
    
    # Step 1
    if args.benchmark:
        logger.set_memory()
    classfier = DeepClassifier()
    trainset, testset = classfier.init(args)
    classfier.train(trainset, load_saved=True)
    
    # Step 2
    if args.benchmark:
        logger.set_memory()
        logger.start_measure_memory(0.0001)
        logger.start_measure_time("test")
    y_true, y_pred = classfier.test(testset)
    
    # Step 3
    if args.benchmark:
        logger.end_measure_time("test")
        logger.end_measure_memory_and_terminate()
        # del classfier
        # del testset
        # gc.collect()
    
    if not args.benchmark:
        # acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        bacc = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        # conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        
        logger.write_accuracy("bacc", bacc)
        logger.write_accuracy("conf_mat", metrics.confusion_matrix(y_true=y_true, y_pred=y_pred))
        # print(metrics.classification_report(y_true, y_pred))
        print(f"bacc: {bacc}")


# @profile
def run_shallowClassifier(args):
    np.random.seed(args.seed)
    
    logger = ExpLogger(
        "../results/journal",
        exp_name=f"{args.exp_name}",
        tag=f"{args.dataset}_{args.n_shots}_{args.model}_{args.benchmark}_{args.seed}",
        default_overwrite=args.overwrite,
        log_start_time=False,
        log_accuracy=not args.benchmark,
        log_memory=args.benchmark,
        log_time=False
    )
    
    if logger.terminate == True:
        return
    
    # Step 1
    if args.benchmark:
        logger.set_memory()
    
   
    classfier = ShallowClassifier()
    trainset, testset = classfier.init(args)
    classfier.train(trainset, load_saved=True)
    
    # Step 2
    if args.benchmark:
        logger.set_memory()
        logger.start_measure_memory(0.0001)
        logger.start_measure_time("test")
    y_true, y_pred = classfier.test(testset)
    
    # Step 3
    if args.benchmark:
        logger.end_measure_time("test")
        time.sleep(0.1)
        logger.end_measure_memory_and_terminate()
        # del classfier
        # del testset
        # gc.collect()
    
    if not args.benchmark:
        # acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        bacc = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        # conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        
        logger.write_accuracy("bacc", bacc)
        logger.write_accuracy("conf_mat", metrics.confusion_matrix(y_true=y_true, y_pred=y_pred))
        # print(metrics.classification_report(y_true, y_pred))
        print(f"bacc: {bacc}", flush=True)

# @profile
def run_gzipClassifier(args):
    np.random.seed(args.seed)

    logger = ExpLogger(
        "../results/journal",
        exp_name=f"{args.exp_name}",
        tag=f"{args.dataset}_{args.n_shots}_{args.decimal}_{args.k}_{args.method}_{args.benchmark}_{args.seed}",
        default_overwrite=args.overwrite,
        log_start_time=False,
        log_accuracy=not args.benchmark,
        log_memory=args.benchmark,
        log_time=False
    )

    if logger.terminate == True:
        return

    # Step 1
    if args.benchmark:
        logger.set_memory()
    
    classfier = GzipClassifier()
    trainset, testset = classfier.init(args)
    
    # Step 2
    if args.benchmark:
        logger.set_memory()
        logger.start_measure_memory(0.0001)
        logger.start_measure_time("all")
    y_true, y_pred = classfier.run(trainset, testset)
    
    # Step 3
    if args.benchmark:
        logger.end_measure_time("all")
        time.sleep(0.1)
        logger.end_measure_memory_and_terminate()
        # del classfier
        # del testset
        # gc.collect()
    
    if not args.benchmark:
        # if benchmark, acc is bullshit.
        
        # acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        bacc = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        # conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        
        logger.write_accuracy("bacc", bacc)
        logger.write_accuracy("conf_mat", metrics.confusion_matrix(y_true=y_true, y_pred=y_pred))
        # print(metrics.classification_report(y_true, y_pred))
        print(f"bacc: {bacc}")

if __name__ == "__main__":
    args = parse_args()
    
    eval(f"run_{args.scheme}Classifier")(args)