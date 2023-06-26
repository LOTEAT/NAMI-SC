'''
Author: LOTEAT
Date: 2023-06-20 19:30:25
'''
import torch
from torch import nn
from nltk.translate.bleu_score import sentence_bleu
from w3lib.html import remove_tags
from transformers import BertModel, BertTokenizer

import torch.nn.functional as F


def SparseCategoricalCrossentropyLoss(real, pred, ignore_index=0):
    loss_object = nn.CrossEntropyLoss(reduction='none')
    mask = real != ignore_index
    bs = pred.shape[0]
    loss_ = loss_object(pred.view(-1, 22234), real.contiguous().view(-1))
    loss_ = loss_.view(bs, -1)
    loss_ *= mask.float()
    return torch.mean(loss_)
    
    



def bleu_score(targets, predicts, weights=(1, 0, 0, 0)):
    get_score = lambda target, predict: sentence_bleu(target, predict, weights=weights)
    scores = [get_score([remove_tags(target).split()], remove_tags(predict).split()) for target, predict in zip(targets, predicts)]
    return scores


def sentence_similarity(targets, predicts):
    bert_model = BertModel.from_pretrained()
    tokenizer = BertTokenizer.from_pretrained()


# class Similarity():
#     def __init__(self, config_path, checkpoint_path, dict_path):
#         self.model1 = BertModel.from_pretrained(checkpoint_path)
#         self.model = nn.Sequential(*list(self.model1.children())[:-1])
#         self.tokenizer = BertTokenizer.from_pretrained(dict_path, do_lower_case=True)

#     def compute_score(self, real, predicted):
#         token_ids1, segment_ids1 = [], []
#         token_ids2, segment_ids2 = [], []
#         score = []

#         for (sent1, sent2) in zip(real, predicted):
#             sent1 = remove_tags(sent1)
#             sent2 = remove_tags(sent2)

#             inputs1 = self.tokenizer.encode_plus(sent1, add_special_tokens=True, max_length=32, truncation=True)
#             inputs2 = self.tokenizer.encode_plus(sent2, add_special_tokens=True, max_length=32, truncation=True)

#             token_ids1.append(inputs1['input_ids'])
#             token_ids2.append(inputs2['input_ids'])
#             segment_ids1.append(inputs1['token_type_ids'])
#             segment_ids2.append(inputs2['token_type_ids'])

#         token_ids1 = torch.tensor(token_ids1)
#         token_ids2 = torch.tensor(token_ids2)
#         segment_ids1 = torch.tensor(segment_ids1)
#         segment_ids2 = torch.tensor(segment_ids2)

#         vector1 = self.model(token_ids1, segment_ids1)[0]
#         vector2 = self.model(token_ids2, segment_ids2)[0]

#         vector1 = torch.sum(vector1, dim=1)
#         vector2 = torch.sum(vector2, dim=1)

#         vector1 = normalize(vector1.detach().numpy(), axis=0, norm='max')
#         vector2 = normalize(vector2.detach().numpy(), axis=0, norm='max')

#         dot = np.diag(np.matmul(vector1, vector2.T))
#         a = np.diag(np.matmul(vector1, vector1.T))
#         b = np.diag(np.matmul(vector2, vector2.T))

#         a = np.sqrt(a)
#         b = np.sqrt(b)

#         output = dot / (a * b)
#         score = output.tolist()

        # return score


