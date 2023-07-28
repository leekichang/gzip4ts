import os
import argparse
import pickle

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    # default arguments
    parser.add_argument('--exp_name', help='quant_str'         , type=str     , required=True)
    parser.add_argument('--dataset' , help='dataset'           , type=str     , default='mitbih_arr')
    parser.add_argument('--seed'    , help='random seed'       , type=int     , default=6206)
    parser.add_argument('--decimal' , help='decimal point'     , type=int     , default=4)
    parser.add_argument('--n_shots' , help='number of knn data', type=int     , default=5)
    parser.add_argument('--k'       , help='top-k for knn'     , type=int     , default=3)
    parser.add_argument('--dtype'   , help='data type'         , type=str     , default='str')
    parser.add_argument('--save'    , help='save exp result?'  , type=str2bool, default=True)
    args = parser.parse_args()
    return args

def check_path(path):
    os.makedirs(path, exist_ok=True)

def save_result(args, y_true, y_pred, save_path='./exp_results'):
    check_path(save_path)
    info = {'exp_name': args.exp_name,
            'seed': args.seed,
            'decimal': args.decimal,
            'n_shots': args.n_shots,
            'dtype': args.dtype,
            'y_true': y_true,
            'y_pred': y_pred}
    with open(f'{save_path}/{args.exp_name}_{args.decimal}_{args.n_shots}_{args.k}_{args.dtype}.pickle', 'wb') as f:
        pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

def load_result(args, save_path='./exp_results'):
    with open(f'{save_path}/{args.exp_name}_{args.decimal}_{args.n_shots}_{args.k}_{args.dtype}.pickle', 'rb') as f:
        info = pickle.load(f)
    #print(info)
    
def class_repr(class_idx):
    return ["LBBB", 'N', 'PB', 'PVC','RBBB','VF'][int(class_idx)]