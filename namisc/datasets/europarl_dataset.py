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
import json
# from .load_data import load_data, load_rays

@DATASETS.register_module()
class EuroparlDataset(BaseDataset):    
    def __init__(self, cfg, pipeline):
        super().__init__()
        self.iter_n = 0
        self.mode = cfg.mode
        self.cfg = cfg
        data = pickle.load(open(cfg.path, 'rb'))[:1000]
        self.vocab = json.load(open(cfg.vocab_path, 'rb'))
        token_to_idx = self.vocab['token_to_idx']
        self.vocab = token_to_idx
        self.start_idx = token_to_idx["<START>"]
        self.end_idx = token_to_idx["<END>"]
        self.pad_idx = token_to_idx["<PAD>"]        
        self.data = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(seq) for seq in data], batch_first=True)
        self._init_pipeline(pipeline)

    def get_info(self):
        res = {
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'pad_idx': self.pad_idx
        }
        return res

    def _fetch_train_data(self, idx):
        data = {
            'data': self.data,
            'idx': idx,
        }
        return data

    def _fetch_val_data(self, idx):  
        data = {
            'data': self.data,
            'idx': idx
        }
        return data

    def _fetch_test_data(self, idx): 
        data = {
            'data': self.data,
            'idx': idx
        }
        return data

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self._fetch_train_data(idx)
            data = self.pipeline(data)
            return data
        elif self.mode == 'val':  # for some complex reasons，pipeline have to be moved to network.val_step() in val phase
            data = self._fetch_val_data(idx)
            data = self.pipeline(data)
            return data
        elif self.mode == 'test':  # for some complex reasons，pipeline have to be moved to network.val_step() in test phase
            data = self._fetch_test_data(idx)
            data = self.pipeline(data)
            return data

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        elif self.mode == 'val':
            return len(self.data)
        elif self.mode == 'test':
            return len(self.data)
        
    def extra_func(self):
        def token2text(tokens, vocab, end_idx):
            reverse_word_map = dict(zip(vocab.values(), vocab.keys()))
            words = []
            for token in tokens:
                if token == end_idx:
                    break
                else:
                    words.append(reverse_word_map.get(token))
            words = ' '.join(words)
            return words
        return lambda tokens: token2text(tokens, self.vocab, self.end_idx)
    
    def extra_data(self):
        return None

