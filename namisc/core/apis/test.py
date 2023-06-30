'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''
from mmcv.parallel import MMDataParallel
from namisc.models.builder import build_transceiver
from namisc.utils import get_root_logger
from .helper import build_dataloader, get_runner, register_hooks


def test_sc(cfg):
    """test model entry function.

    Args:
        cfg (dict): The config dict for test, the same config as train.
        the difference between test and val is:
                    in test phase, use 'EpochBasedRunner' to influence all testset, in one iter
                    in val phase, use 'IterBasedRunner' to influence 1/N testset, in one epoch (several iters)
    """
    cfg.workflow = [('val', 1)]  # only run val_step one epoch

    extra_info = {}
    test_loader, testset = build_dataloader(cfg, mode='test')
    
    extra_info['extra_func'] = testset.extra_func()
    extra_info['extra_data'] = testset.extra_data()
    
    dataloaders = [test_loader]

    network = build_transceiver(cfg.model)

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
    network = MMDataParallel(network.cuda(), device_ids=[])

    Runner = get_runner(cfg.test_runner)
    runner = Runner(network,
                    work_dir=cfg.work_dir,
                    logger=get_root_logger(log_level=cfg.log_level),
                    meta=None)
    runner.timestamp = cfg.get('timestamp', None)
    register_hooks(cfg.test_hooks, **locals())

    runner.load_checkpoint(cfg.load_from)

    print('start test...', flush=True)
    
    runner_kwargs = extra_info
    
    runner.run(data_loaders=dataloaders, workflow=cfg.workflow, max_epochs=cfg.test_epochs, **runner_kwargs)
