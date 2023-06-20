_base_ = [
]

import os
from datetime import datetime

method = 'deepsc' 

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

max_epochs = 100
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
    se=dict(  
        type='DeepSCSemanticEncoder',
        num_layers=4,
        num_heads=8,
        d_model=128,
        dff=512,
        vocab_size=22234
    ),
    ce=dict(
        type='DeepSCChannelEncoder'
    ),
    channel=dict(
        type='Awgn',
    ),
    cd=dict( 
        type='DeepSCChannelDecoder',
    ),
    sd=dict( 
        type='DeepSCSemanticDecoder',
        num_layers=4,
        num_heads=8,
        d_model=128,
        dff=512,
        vocab_size=22234
    ),
)

traindata_cfg = dict(
    datadir='data/#DATANAME#',
    mode='train',
    path='data/#DATANAME#/train_data.pkl',
)
valdata_cfg = dict(
    datadir='data/#DATANAME#',
    mode='val',
    path='data/#DATANAME#/test_data.pkl',
)
testdata_cfg = dict(
    datadir='data/#DATANAME#',
    mode='test',
    path='data/#DATANAME#/test_data.pkl',
)

traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val'))
testdata_cfg.update(dict(mode='test', testskip=0))

train_pipeline = [
    dict(
        type='ToTensor',
        enable=True,
        keys=['data'],
    ),
    dict(
        type='Sample',
        enable=True,
    ),
]

test_pipeline = [
]

data = dict(
    train_loader=dict(batch_size=64, num_workers=0),
    train=dict(
        type='EuroparlDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=64, num_workers=0),
    val=dict(
        type='EuroparlDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ),
    test_loader=dict(batch_size=64, num_workers=0),
    test=dict(
        type='EuroparlDataset',
        cfg=testdata_cfg,
        pipeline=test_pipeline,  # same pipeline as validation
    ),
)
