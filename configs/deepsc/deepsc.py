_base_ = [
]

import os
from datetime import datetime

method = 'deepsc' 

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

max_iters = 20
lr_config = dict(policy='step', step=500 * 1000, gamma=0.1, by_epoch=False)
checkpoint_config = dict(interval=5, by_epoch=False)
log_level = 'INFO'
log_config = dict(interval=5,
                  by_epoch=False,
                  hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 5), ('val', 1)]

# hooks
# 'params' are numeric type value, 'variables' are variables in local environment
train_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='valset')),
    dict(type='ValidateHook',
         params=dict(save_folder='visualizations/validation')),
    dict(type='SaveSpiralHook',
         params=dict(save_folder='visualizations/spiral')),
    dict(type='PassIterHook', params=dict()),  # 将当前iter数告诉dataset
    dict(type='OccupationHook',
         params=dict()),  # no need for open-source vision
    # dict(type='SaveDistillResultsHook', params=dict(), variables=dict(model='network', cfg='cfg', trainset='trainset')), # kilo示例
]

test_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='testset')),
    dict(type='TestHook', params=dict()),
]

# runner
train_runner = dict(type='DeepSCTrainRunner')
test_runner = dict(type='DeepSCTestRunner')

# runtime settings
num_gpus = 1
distributed = (num_gpus > 1)  
work_dir = './work_dirs/deepsc/deepsc_#DATANAME#/'
timestamp = datetime.now().strftime('%d-%b-%H-%M')

# resume_from = os.path.join(work_dir, 'latest.pth')
load_from = os.path.join(work_dir, 'latest.pth')

model = dict(
    type='DeepSCTranseiver',
    cfg=dict(
        phase='train',  # 'train' or 'test'
    ),
    mlp=dict(  # coarse model
        type='NerfMLP',
    ),
    mlp_fine=dict(  # fine model
        type='NerfMLP',
    ),
    render=dict(  # render model
        type='NerfRender',
    ),
)

basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir='data/nerf_llff_data/#DATANAME#',
    half_res=False,  # load blender synthetic data at 400x400 instead of 800x800
    testskip=
    8,  # will load 1/N images from test/val sets, useful for large datasets like deepvoxels
    N_rand_per_sampler=N_rand_per_sampler,
    llffhold=8,  # will take every 1/N images as LLFF test set, paper uses 8
    no_ndc=no_ndc,
    white_bkgd=white_bkgd,
    spherify=False,  # set for spherical 360 scenes
    shape='greek',  # options : armchair / cube / greek / vase
    factor=8,  # downsample factor for LLFF images
    is_batching=True,  # True for blender, False for llff
    mode='train',
)

traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()
testdata_cfg = basedata_cfg.copy()

traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val'))
testdata_cfg.update(dict(mode='test', testskip=0))

train_pipeline = [
    dict(
        type='BatchSample',
        enable=True,
        N_rand=N_rand_per_sampler,
    ),
    dict(type='DeleteUseless', keys=['rays_rgb', 'idx']),
    dict(
        type='ToTensor',
        enable=True,
        keys=['rays_o', 'rays_d', 'target_s'],
    ),
    dict(
        type='GetViewdirs',
        enable=use_viewdirs,
    ),
    dict(
        type='ToNDC',
        enable=(not no_ndc),
    ),
    dict(type='GetBounds', enable=True),
    dict(type='GetZvals', enable=True, lindisp=lindisp, N_samples=N_samples),
    dict(type='PerturbZvals', enable=is_perturb),
    dict(type='GetPts', enable=True),
    dict(type='DeleteUseless', enable=True, keys=['iter_n']),  # iter_n
]

test_pipeline = [
    dict(
        type='ToTensor',
        enable=True,
        keys=['pose'],
    ),
    dict(
        type='GetRays',
        enable=True,
    ),
    dict(type='FlattenRays',
         enable=True),  # 原来是(H, W, ..) 变成(H*W, ...) 记录下原来的尺寸
    dict(
        type='GetViewdirs',
        enable=use_viewdirs,
    ),
    dict(
        type='ToNDC',
        enable=(not no_ndc),
    ),
    dict(type='GetBounds', enable=True),
    dict(type='GetZvals', enable=True, lindisp=lindisp,
         N_samples=N_samples),  # 同上train_pipeline
    dict(type='PerturbZvals', enable=False),  # 测试集不扰动
    dict(type='GetPts', enable=True),
    dict(type='DeleteUseless', enable=True,
         keys=['pose']),  # 删除pose 其实求完ray就不再需要了
]

data = dict(
    train_loader=dict(batch_size=4, num_workers=4),
    train=dict(
        type='SceneBaseDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='SceneBaseDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ),
    test_loader=dict(batch_size=1, num_workers=0),
    test=dict(
        type='SceneBaseDataset',
        cfg=testdata_cfg,
        pipeline=test_pipeline,  # same pipeline as validation
    ),
)
