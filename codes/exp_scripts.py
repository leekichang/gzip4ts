import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import itertools
from multiprocessing import cpu_count
from threading import Lock
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
from memory_profiler import profile

from classifiers.shallow_classifier import MLTrainer as ShallowClassifier
from main import run_shallowClassifier, run_gzipClassifier
from utils import ExpLogger, arguments
import subprocess

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

def exp1_rungzip_variousparam_0704():
    """
    Accuracy is precise
    others are not precise
    """
    
    exp_name = "exp1_rungzip_variousparam_0724"
    
    datasets = ["mitbih_arr", "pamap2", "mitbih_auth"] # ["mitbih_arr", "pamap2", "mitbih_auth"]
    n_shots = [1, 2, 3, 4, 8, 16, 32] # [1, 2, 3, 4, 8, 16, 32]
    ks = [1, 3, 5, 7, 9, 21] # [1, 3, 5, 7, 9, 21]
    decimals = [1, 2, 3, 4, 5, 6, 7, 8, 16] # [ 1, 2, 3, 4, 5, 6, 7, 8 ,16]
    seeds = range(3)
    methods = ["all"] # ["default", "all"]
    
    combinations = list(itertools.product(datasets, n_shots, methods, decimals, ks, seeds))
    
    # Only good models
    combinations = [
        # dataset, n_shot, decimal, k
        ['mitbih_arr', 1, 1, 1, 0],
    ]
    method = "all"
    seeds = range(3, 10)
    
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
        
        # if decimal is not None and decimal > 5:
        #     print(f"dataset:{dataset}, n_shot:{n_shot}, method:{method}, decimal:{decimal}, k:{k}, seed:{seed}")
        
        run_gzipClassifier(args)


def exp5_rungzip_specific_0707():
    
    exp_name = "exp5_rungzip_specific_0710"
    
    # Only good models
    combinations = [
        # dataset, n_shot, decimal, k
        # ['mitbih_arr',  1,  6, 1],
        # ['mitbih_arr',  3,  2, 1],
        # ['mitbih_arr',  16, 4, 1],
        ['mitbih_arr',  32, 4, 1],
        # ['mitbih_auth', 1,  3, 1],
        # ['mitbih_auth', 8,  3, 1],
        ['mitbih_auth', 16, 2, 1],
        # ['pamap2',      1,  2, 1],
        # ['pamap2',      3,  2, 1],
        ['pamap2',      32, 2, 1],
    ]
    method = "all"
    seeds = range(300)
    
    combinations = itertools.product(seeds, combinations)
    combinations = [(seed, *comb) for seed, comb in combinations]
    
    overwrite=True
    # # acc
    # for seed, dataset, n_shot, decimal, k in tqdm(combinations):
    #     args = argparse.Namespace(
    #         exp_name=exp_name,
    #         dataset=dataset,
    #         seed=seed,
    #         n_shots=n_shot,
    #         decimal=decimal,
    #         k=k,
    #         method=method,
    #         benchmark=False,
    #         overwrite=overwrite
    #     )
        
    #     commands = ["python", "main.py", f"--scheme=gzip", f"--exp_name={args.exp_name}", f"--dataset={args.dataset}", f"--seed={args.seed}", f"--n_shots={args.n_shots}", f"--decimal={args.decimal}", f"--k={args.k}", f"--method={args.method}", f"--benchmark={args.benchmark}", f"--overwrite={args.overwrite}"]
    #     # print(" ".join(commands))
    #     subprocess.run(commands)

    # benchmark
    with tqdm(total=len(combinations)) as pbar:
        with ThreadPoolExecutor(max_workers=(cpu_count()-4)//2) as executor:
            for seed, dataset, n_shot, decimal, k in tqdm(combinations):
                args = argparse.Namespace(
                    exp_name=exp_name,
                    dataset=dataset,
                    seed=seed,
                    n_shots=n_shot,
                    decimal=decimal,
                    k=k,
                    method=method,
                    benchmark=True,
                    overwrite=overwrite
                )
                
                command = ["python", "main.py", f"--scheme=gzip", f"--exp_name={args.exp_name}", f"--dataset={args.dataset}", f"--seed={args.seed}", f"--n_shots={args.n_shots}", f"--decimal={args.decimal}", f"--k={args.k}", f"--method={args.method}", f"--benchmark={args.benchmark}", f"--overwrite={args.overwrite}"]
                print(" ".join(command))
                fut = executor.submit(subprocess.run, command)
                fut.add_done_callback(lambda x: pbar.update(1))

def _exp3_runprocess(args):
    commands = ["python", "main.py", f"--scheme=gzip", f"--exp_name={args.exp_name}", f"--dataset={args.dataset}", f"--seed={args.seed}", f"--n_shots={args.n_shots}", f"--decimal={args.decimal}", f"--k={args.k}", f"--method={args.method}", f"--benchmark={args.benchmark}", f"--overwrite={args.overwrite}"]
        # print(" ".join(commands))
    subprocess.run(commands)

def exp3_rungzip_benchmark_0706():
    exp_name = "exp3_rungzip_benchmark_0710"
    
    # # mitbih_auth_1_3_1_all_True_1 
    # args = argparse.Namespace(
    #     exp_name=exp_name,
    #     dataset="mitbih_auth",
    #     seed=2,
    #     n_shots=1,
    #     decimal=3,
    #     k=1,
    #     method="all",
    #     benchmark=True
    # )
    
    # run_gzipClassifier(args)
    
    # exit()
    
    
    datasets = ["mitbih_arr", "pamap2", "mitbih_auth"] # ["mitbih_arr", "pamap2", "mitbih_auth"]
    n_shots = [1, 2, 3, 4, 8, 16, 32] # [1, 2, 3, 4, 8, 16, 32, 64]
    ks = [1] # [1, 3, 5, 7, 9, 21]
    
    # (dataset, n_shot): (decimal)
    decimals = {
        "mitbih_arr": {
            1: [6],
            2: [2],
            3: [2],
            4: [4],
            8: [4],
            16: [4],
            32: [4]
        },
        "mitbih_auth": {
            1: [3],
            2: [3],
            3: [3],
            4: [3],
            8: [3],
            16: [2],
            32: [2]
        },
        "pamap2": {
            1: [2],
            2: [2],
            3: [2],
            4: [2],
            8: [2],
            16: [2],
            32: [2]
        },
    }
    
    seeds = range(10)
    methods = ["all"] # ["default", "all"]
    
    # datasets = ["mitbih_arr"] # ["mitbih_arr", "pamap2", "mitbih_auth"]
    # n_shots = [1, 8, 32] # [1, 2, 3, 4, 8, 16, 32, 64]
    # ks = [1] # [1, 3, 5, 7, 9, 21]
    # seeds = range(1)
    
    combinations = list(itertools.product(seeds, datasets, n_shots, methods, ks))
    overwrite = True
    with tqdm(total=len(combinations)) as pbar:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for i, (seed, dataset, n_shot, method, k) in enumerate(tqdm(combinations)):
                
                for decimal in decimals[dataset][n_shot]:
                        
                    args = argparse.Namespace(
                        exp_name=exp_name,
                        dataset=dataset,
                        seed=seed,
                        n_shots=n_shot,
                        decimal=decimal,
                        k=k,
                        method=method,
                        benchmark=True,
                        overwrite=overwrite
                    )
                    
                    if not args.overwrite:
                        if os.path.exists(f"../results/journal/{exp_name}/_{dataset}_{n_shot}_{decimal}_{k}_{method}_True_{seed}"):
                            continue
                    
                    # fut = executor.submit(_exp3_runprocess, args)
                    # fut.add_done_callback(lambda x: pbar.update(1))
                    # # run_gzipClassifier(args, default_overwrite=(3<=i<3+3))
                    
                    command = ["python", "main.py", f"--scheme=gzip", f"--exp_name={args.exp_name}", f"--dataset={args.dataset}", f"--seed={args.seed}", f"--n_shots={args.n_shots}", f"--decimal={args.decimal}", f"--k={args.k}", f"--method={args.method}", f"--benchmark={args.benchmark}", f"--overwrite={args.overwrite}"]
                    # fut = executor.submit(
                    #     lambda x: subprocess.run(command),
                    #     args
                    # )
                    # fut.add_done_callback(lambda x: pbar.update(1))
                    
                    subprocess.run(command)
                    print(" ".join(command))
                    pbar.update(1)

def exp2_runshallow_acc_0705():
    
    exp_name = "exp2_runshallow_acc_0707"
    datasets = ["mitbih_arr", "pamap2", "mitbih_auth"]
    # datasets = ["mitbih_arr", "pamap2"]
    n_shots = [1, 2, 3, 4, 8, 16, 32] # [1, 2, 3, 4, 8, 16, 32, 64]
    seeds = range(10)
    # models = ['SVC', 'DT', 'RF', 'GP', 'AB', 'MLP']
    models = ['DT']
    
    combinations = list(itertools.product(seeds, models, datasets, n_shots))
    
    from main import run_shallowClassifier
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import cpu_count
    with tqdm(total=len(combinations)) as pbar:
        with ProcessPoolExecutor(max_workers=cpu_count()-2) as executor:
            for seed, model, dataset, n_shot in combinations:
                args = argparse.Namespace(
                    exp_name=exp_name,
                    dataset=dataset,
                    seed=seed,
                    n_shots=n_shot,
                    model=model,
                    benchmark=False,
                    overwrite=True
                )
                
                # if not args.overwrite:
                #     if os.path.exists(f"../results/journal/{exp_name}/_{dataset}_{n_shot}_{model}_True_{seed}"):
                #         continue
                
                fut = executor.submit(run_shallowClassifier, args)
                fut.add_done_callback(lambda x: pbar.update(1))
 
def exp4_runshallow_benchmark_0706():
    
    exp_name = "exp4_runshallow_benchmark_0710"
    datasets = ["mitbih_arr", "pamap2", "mitbih_auth"]
    # datasets = ["mitbih_arr", "pamap2"]
    n_shots = [1, 2, 3, 4, 8, 16, 32] # [1, 2, 3, 4, 8, 16, 32, 64]
    seeds = range(10)
    models = ['SVC', 'DT', 'RF', 'GP', 'AB', 'MLP'] 
    # models = ['DT']
    
    combinations = list(itertools.product(seeds, models, datasets, n_shots))
    
    overwrite = True
    with tqdm(total=len(combinations)) as pbar:
        with ThreadPoolExecutor(max_workers=(cpu_count()-4)//2) as executor:
            for seed, model, dataset, n_shot in (combinations):
                
                args = argparse.Namespace(
                    exp_name=exp_name,
                    dataset=dataset,
                    seed=seed,
                    n_shots=n_shot,
                    model=model,
                    benchmark=True,
                    overwrite=overwrite
                )
                
                if not args.overwrite:
                    if os.path.exists(f"../results/journal/{exp_name}/_{dataset}_{n_shot}_{model}_True_{seed}"):
                        continue
                
                commands = ["python", "main.py", f"--scheme=shallow", f"--exp_name={args.exp_name}", f"--dataset={args.dataset}", f"--seed={args.seed}", f"--n_shots={args.n_shots}", f"--model={args.model}", f"--benchmark={args.benchmark}", f"--overwrite={args.overwrite}"]
                print(" ".join(commands))
                fut = executor.submit(subprocess.run, commands)
                fut.add_done_callback(lambda x: pbar.update(1))

def exp6_rundeep_acc_0708():
    def manage_gpu_availability(gpu_id):
        with lock:
            time.sleep(1)
            torch.cuda.empty_cache()
            gpu_is_available[gpu_id] = True
        
        
    exp_name = "exp6_rundeep_acc_0708"
    datasets = ["mitbih_arr", "pamap2", "mitbih_auth"]
    n_shots = [1, 2, 3, 4, 8, 16, 32] # [1, 2, 3, 4, 8, 16, 32, 64]
    seeds = range(10)
    models = ['CNN', 'ResNet', 'GRU', 'LSTM']
    # models = ['CNN', 'ResNet', 'GRU']
    combinations = list(itertools.product(seeds, models, datasets, n_shots))
    
    from main import run_deepClassifier
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import cpu_count
    
    # available_gpus = {
    #     "cuda:0": True,
    #     "cuda:1": True,
    #     "cuda:2": True,
    #     "cuda:3": True,
    # }
    num_gpus = 4  

    
    gpu_is_available = {
        i: True for i in range(num_gpus)
    }
    
    lock = Lock()
    
    with tqdm(total=len(combinations)) as pbar:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = {}
            for i, (seed, model, dataset, n_shot) in enumerate(combinations):
                
                if model in ['CNN', 'ResNet']:
                    batch_size, epochs = 2048, 50
                elif model == 'GRU':
                    batch_size, epochs = 512, 200
                elif model == 'LSTM':
                    batch_size, epochs = 512 if n_shot < 16 else 256 if n_shot < 32 else 128 if dataset != "mitbih_arr" else 64, 200
                
                args = argparse.Namespace(
                    exp_name=exp_name,
                    dataset=dataset,
                    seed=seed,
                    n_shots=n_shot,
                    model=model,
                    
                    batch_size=batch_size,
                    epochs=epochs,
                    
                    use_tb=True,
                    benchmark=False,
                    overwrite=False
                )
                
                if not args.overwrite:
                    if os.path.exists(f"../results/journal/{exp_name}/_{dataset}_{n_shot}_{model}_{args.epochs}_{args.batch_size}_False_{seed}"):
                        pbar.update(1)
                        continue
                
                available_gpu = None
                while available_gpu is None:
                    with lock:
                        for gpu_id, is_available in gpu_is_available.items():
                            if is_available:
                                available_gpu = gpu_id
                                gpu_is_available[available_gpu] = False
                                break
                    time.sleep(0.1)
                args.device=f"cuda:{available_gpu}"
                fut = executor.submit(run_deepClassifier, args)
                
                fut.add_done_callback(lambda x, gpu_id=available_gpu: manage_gpu_availability(gpu_id))
                fut.add_done_callback(lambda x: pbar.update(1))

def exp7_rundeep_benchmark_0708():
        
    exp_name = "exp7_rundeep_benchmark_0710"
    datasets = ["mitbih_arr", "pamap2", "mitbih_auth"]
    n_shots = [1] # [1, 2, 3, 4, 8, 16, 32, 64]
    seeds = range(10)
    models = ['CNN', 'ResNet', 'GRU', 'LSTM']
    # models = ['LSTM']
    combinations = list(itertools.product(seeds, models, datasets, n_shots))
    
    overwrite = True
    with tqdm(total=len(combinations)) as pbar:
        with ThreadPoolExecutor(max_workers=(cpu_count()-4)//2) as executor:
            for i, (seed, model, dataset, n_shot) in enumerate(combinations):
                
                if model in ['CNN', 'ResNet']:
                    batch_size, epochs = 2048, 50
                elif model == 'GRU':
                    batch_size, epochs = 512, 200
                elif model == 'LSTM':
                    batch_size, epochs = 512 if n_shot < 16 else 256 if n_shot < 32 else 128 if dataset != "mitbih_arr" else 64, 200
                        
                        
                args = argparse.Namespace(
                    exp_name=exp_name,
                    dataset=dataset,
                    seed=seed,
                    n_shots=n_shot,
                    model=model,
                    
                    batch_size=batch_size,
                    epochs=epochs,
                    
                    device=f"cpu",
                    use_tb=False,
                    benchmark=True,
                    overwrite=overwrite
                )
                
                # if not args.overwrite:
                #     if os.path.exists(f"../results/journal/{exp_name}/_{dataset}_{n_shot}_{model}_{args.epochs}_{args.batch_size}_True_{seed}"):
                #         continue
                
            
                commands = ["python", "main.py", f"--scheme=deep", 
                            f"--exp_name={args.exp_name}", f"--dataset={args.dataset}", 
                            f"--seed={args.seed}", f"--n_shots={args.n_shots}",
                            f"--model={args.model}", f"--epochs={args.epochs}", f"--batch_size={args.batch_size}",
                            f"--benchmark={args.benchmark}", f"--overwrite={args.overwrite}", f"--use_tb={args.use_tb}"
                ]
                print(" ".join(commands))
                fut = executor.submit(subprocess.run, commands)
                fut.add_done_callback(lambda x: pbar.update(1))
                # subprocess.run(commands)
                # pbar.update(1)

def exp9_rungzip_ablation_0709():
        
        exp_name = "exp9_rungzip_ablation_0709"
        
        # Only good models
        combinations = [
            # dataset, n_shot, decimal, k
            # ['mitbih_arr',  32, 4, 1],
            # ['mitbih_auth', 16, 2, 1],
            ['pamap2', 32, 2, 1],
        ]
        methods = ['default', 'fpq', 'hybrid', 'cw']
        seeds = range(300)
        
        combinations = itertools.product(seeds, methods, combinations)
        combinations = [(seed, method, *comb) for seed, method, comb in combinations]
        print(combinations)
        overwrite = True
        # with tqdm(total=len(combinations)) as pbar:
        #     for seed, method, dataset, n_shot, decimal, k in tqdm(combinations):
        #         args = argparse.Namespace(
        #             exp_name=exp_name,
        #             dataset=dataset,
        #             seed=seed,
        #             n_shots=n_shot,
        #             decimal=decimal,
        #             k=k,
        #             method=method,
        #             benchmark=False,
        #             overwrite=overwrite
        #         )
                
        #         run_gzipClassifier(args)
        #         pbar.update(1)
        
        with tqdm(total=len(combinations)) as pbar:
            with ThreadPoolExecutor(max_workers=(cpu_count()-4)//2) as executor:
                for seed, method, dataset, n_shot, decimal, k in tqdm(combinations):
                    args = argparse.Namespace(
                        exp_name=exp_name,
                        dataset=dataset,
                        seed=seed,
                        n_shots=n_shot,
                        decimal=decimal,
                        k=k,
                        method=method,
                        benchmark=True,
                        overwrite=overwrite
                    )
                    
                    commands = ["python", "main.py", f"--scheme=gzip", f"--exp_name={args.exp_name}", f"--dataset={args.dataset}", f"--seed={args.seed}", f"--n_shots={args.n_shots}", f"--decimal={args.decimal}", f"--k={args.k}", f"--method={args.method}", f"--benchmark={args.benchmark}", f"--overwrite={args.overwrite}"]
                    print(" ".join(commands))
                    fut = executor.submit(subprocess.run, commands)
                    fut.add_done_callback(lambda x: pbar.update(1))
                    
                    # subprocess.run(command)
                    # print(" ".join(command))
                    # pbar.update(1)

if __name__ == '__main__':
    # args = arguments.parse_args()
    
    # import pprint
    # pprint.pprint(args)
    
    # np.random.seed(args.seed)
    
    # exp1_rungzip_variousparam_0704()
    
    # exp2_runshallow_acc_0705()
    exp4_runshallow_benchmark_0706()
    
    # exp3_rungzip_benchmark_0706()
    # exp5_rungzip_specific_0707()
    
    # exp6_rundeep_acc_0708()
    exp7_rundeep_benchmark_0708()
    
    # exp9_rungzip_ablation_0709()