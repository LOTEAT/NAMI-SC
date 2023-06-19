'''
Author: LOTEAT
Date: 2023-06-19 15:26:35
'''
from namisc.core.apis import *
import torch

if __name__ == '__main__':

    args = parse_args()
    torch.set_num_threads(args.num_threads)
    run_sc(args)
