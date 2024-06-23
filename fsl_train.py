import numpy as np

import utils

if __name__ == '__main__':
    args = utils.parse_args()
    print(f'exp_name:{args.exp_name}\nnum shot:{args.n_shots}\ncompress type:{args.dtype}')
    trainset, testset = utils.load_dataset(dataset=args.dataset)
    trainset = utils.choose_trainset(trainset, args)
    print(trainset.X.shape, trainset.Y.shape)
    print(testset.X.shape, testset.Y.shape)