'''
Author: LOTEAT
Date: 2023-05-31 16:34:26
'''
# TODO 
# This is a temporary version, there may be a more elegant solution
# I will modify these codes soon later
import pickle
from .base import BaseDataset
from .builder import DATASETS
import torch
# from .load_data import load_data, load_rays

@DATASETS.register_module()
class EuroparlDataset(BaseDataset):    
    def __init__(self, cfg, pipeline):
        super().__init__()
        self.iter_n = 0
        self.mode = cfg.mode
        self.cfg = cfg
        data = pickle.load(open(cfg.path, 'rb'))
        self.data = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(seq) for seq in data], batch_first=True)
        self._init_pipeline(pipeline)

    def get_info(self):
        res = {}
        return res

    def _fetch_train_data(self, idx):
        data = {
            'data': self.data,
            'target': self.data,
            'idx': idx
        }
        return data

    def _fetch_val_data(self, idx):  
        return {}

    def _fetch_test_data(self, idx): 
        return {}

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self._fetch_train_data(idx)
            data = self.pipeline(data)
            return data
        elif self.mode == 'val':  # for some complex reasons，pipeline have to be moved to network.val_step() in val phase
            return self._fetch_val_data(idx)
        elif self.mode == 'test':  # for some complex reasons，pipeline have to be moved to network.val_step() in test phase
            data = self._fetch_test_data(idx)
            return data

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        elif self.mode == 'val':
            return None
        elif self.mode == 'test':
            return None

