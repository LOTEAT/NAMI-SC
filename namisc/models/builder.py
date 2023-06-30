'''
Author: LOTEAT
Date: 2023-06-17 23:41:21
'''
# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
SE = MODELS
CE = MODELS
SD = MODELS
CD = MODELS
TRANSCEIVER = MODELS
CHANNEL = MODELS
MINE = MODELS

def build_se(cfg):
    """Build semantic encoder."""
    return SE.build(cfg)


def build_ce(cfg):
    """Build channel encoder."""
    return CE.build(cfg)

def build_sd(cfg):
    """Build semantic decoder."""
    return SD.build(cfg)


def build_cd(cfg):
    """Build channel decoder."""
    return CD.build(cfg)

def build_channel(cfg):
    """Build channel decoder."""
    return CHANNEL.build(cfg)


def build_transceiver(cfg):
    """Build transceiver."""
    return TRANSCEIVER.build(cfg)

def build_mine(cfg):
    """Build mine."""
    return MINE.build(cfg)