'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''
from mmcv import Config

from .helper import update_config, update_loadfrom, update_gpuid
from .test import test_sc
from .train import train_sc

__all__ = ['run_sc']


def run_sc(args):
    cfg = Config.fromfile(args.config)
    cfg = update_config(args.dataname, cfg)
    cfg = update_loadfrom(args.load_from, cfg)
    cfg = update_gpuid(args.gpu_id, cfg)
    if args.test_only:
        cfg['model']['cfg']['phase'] = 'test'
        test_sc(cfg)
    else:
        train_sc(cfg)
