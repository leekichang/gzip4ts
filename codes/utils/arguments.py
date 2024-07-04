import argparse

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
    parser.add_argument('--exp_name'  , help='quant_str'         , type=str     , required=True)
    parser.add_argument('--dataset'   , help='dataset'           , type=str     , default='mitbih_arr')
    parser.add_argument('--seed'      , help='random seed'       , type=int     , default=6206)
    parser.add_argument('--save'      , help='save exp result?'  , type=str2bool, default=True)
    parser.add_argument('--n_shots'   , help='n per class data', type=int     , default=5)
    # kNN Arguments
    parser.add_argument('--decimal'   , help='decimal point'     , type=int     , default=4)
    parser.add_argument('--k'         , help='top-k for knn'     , type=int     , default=3)
    parser.add_argument('--method'    , help='method'            , type=str     , default='str', choices=['default', 'fpq', 'hybrid', 'cw', 'all'])
    # Neural Network Trainer Arguments
    parser.add_argument('--model'     , help='model'             , type=str     , default='CNN')
    parser.add_argument('--epochs'    , help='training epoch'    , type=int     , default=50)
    parser.add_argument('--batch_size', help='batch size'        , type=int     , default=120)
    parser.add_argument('--use_tb'    , help='use tensorboard'   , type=str2bool, default=False)
    args = parser.parse_args()
    return args