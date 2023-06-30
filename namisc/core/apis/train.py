'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''
import os
from mmcv.parallel import MMDataParallel

from namisc.models.builder import build_transceiver
from namisc.utils import get_root_logger
from .helper import build_dataloader, get_optimizer, get_runner, register_hooks


def train_sc(cfg):
    """Train model entry function.

    Args:
        cfg (dict): The config dict for training.
    """
    pass 
    train_loader, trainset = build_dataloader(cfg, mode='train')
    val_loader, valset = build_dataloader(cfg, mode='val')
    dataloaders = [train_loader, val_loader]
    
    extra_info = {}
    extra_info['extra_func'] = trainset.extra_func()
    extra_info['extra_data'] = trainset.extra_data()

    tranceiver = build_transceiver(cfg.model)

    optimizer = get_optimizer(tranceiver, cfg)

    # TODO
    # DDP
    # if cfg.distributed:
    #     print('init_dist...', flush=True)
    #     init_dist('slurm', **cfg.get('dist_param', {}))
    #     find_unused_parameters = cfg.get('find_unused_parameters', False)
    #     network = MMDistributedDataParallel(
    #         network.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False,
    #         find_unused_parameters=find_unused_parameters)
    # else:
    tranceiver = MMDataParallel(tranceiver.cuda(), device_ids=[cfg.get('gpu_id')])

    Runner = get_runner(cfg.train_runner)
    runner = Runner(tranceiver,
                    optimizer=optimizer,
                    work_dir=cfg.work_dir,
                    logger=get_root_logger(log_level=cfg.log_level),
                    meta=None)

    runner.timestamp = cfg.get('timestamp', None)

    # register hooks
    print('register hooks...', flush=True)
    custom_hooks = cfg.get('custom_hooks', None)
    print(custom_hooks)
    runner.register_training_hooks(cfg.lr_config,
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   custom_hooks_config=custom_hooks)
    register_hooks(cfg.train_hooks, **locals())

    # resume_from是载入ckpt和runner的训练信息，load_checkpoint只载入ckpt
    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = extra_info

    print('start train...', flush=True)
    runner.run(dataloaders, cfg.workflow, cfg.max_epochs, **runner_kwargs)
