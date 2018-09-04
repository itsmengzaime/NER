# coding: utf-8

import os
import gzip
import pickle

import numpy as np
import tensorflow as tf

from utils.utils import FeatureExtractor, CRFeeder, conll_format, viterbi_decode, viterbi_decode_topk

from model.CRFModel import CRFModel


def atisfold(fold):
    assert fold in range(5)

    fp = gzip.open('data/atis.fold' + str(fold) + '.pkl.gz')
    train_set, valid_set, test_set, dicts = pickle.load(fp, encoding='iso-8859-1')
    return train_set, valid_set, test_set, dicts


def atisfull():
    with open('data/atis.pkl', 'rb') as fp:
        train_set, test_set, dicts = pickle.load(fp, encoding='iso-8859-1')
    return train_set, test_set, dicts


# train_set, valid_set, test_set, dicts = atisfold(1)
train_set, test_set, dicts = atisfull()

w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
idx2w = {w2idx[k]: k for k in w2idx}
idx2la = {la2idx[k]: k for k in la2idx}

train_x, train_ne, train_la = train_set
test_x, test_ne, test_la = test_set


def dump_data(prefix, x, ne, la):
    wlength = 35
    with open('data/%s.data' % prefix, 'w') as fp:
        for sw, se, sl in zip(x, ne, la):
            for a, b, c in zip(sw, se, sl):
                fp.write(idx2w[a].ljust(wlength) + idx2la[c].ljust(wlength) + '\n')
            fp.write('\n')


print('Dump data...')

dump_data('train', train_x, train_ne, train_la)
dump_data('test', test_x, test_ne, test_la)

print('Load data...')

fe = FeatureExtractor()
fe.parse_template('data/template')

template_vocab_dir = 'dev/template.vocabs'
if os.path.exists(template_vocab_dir):
    fe.construct_vocabs_from_file(template_vocab_dir)
else:
    os.mkdir(template_vocab_dir)
    fe.construct_vocabs_from_data('data/train.data')
    fe.save_vocabs(template_vocab_dir)

'''
[train_size, max_length, feat_size]
feat_size: All unigram features
'''
train_feats = fe.extract_features('data/train.data')
test_feats = fe.extract_features('data/test.data')


print('Load model...')

num_classes = len(la2idx.keys())
max_length = max(
    max(map(len, train_la)),
    max(map(len, test_la)),
)

score = np.array([
    [1, 8, 2],
    [3, 4, 3],
    [2, 1, 3],
    [5, 2, 1]
])
trans_param = np.array([
    [1, 2, 3],
    [7, 8, 1],
    [6, 1, 2]
])

a1, b1 = viterbi_decode(
    score,
    trans_param
)

a2, b2 = viterbi_decode_topk(
    score,
    trans_param,
    2
)
