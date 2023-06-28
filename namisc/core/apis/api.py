'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''
from mmcv import Config

from .helper import update_config, update_loadfrom
from .test import test_sc
from .train import train_sc

__all__ = ['run_sc']


def run_sc(args):
    import torch
    seed = 123
    torch.manual_seed(seed)

    # 如果使用GPU，还需设置以下两行代码
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    cfg = Config.fromfile(args.config)
    cfg = update_config(args.dataname, cfg)
    cfg = update_loadfrom(args.load_from, cfg)
    if args.test_only:
        cfg['model']['cfg']['phase'] = 'test'
        test_sc(cfg)
    else:
        train_sc(cfg)
