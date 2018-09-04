# -*- coding: utf-8 -*-

import gzip
import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec


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

sentences = [
    list(map(lambda x: idx2w[x], sw))
    for sw in train_x
]

emb_size = 128

print("开始训练................")
model = Word2Vec(sentences=sentences, size=emb_size, window=5, min_count=2, workers=8)
print("训练完成!")

print("保存训练结果...........\n")
wv = model.wv

vocab = wv.index2word
vectors = wv.vectors

perm = [w2idx[w] for w in vocab]
perm_reverse = {id2:id1 for id1, id2 in enumerate(perm)}
perm = [perm_reverse[i] for i in range(len(vocab))]

embedding = vectors[perm, :]

np.save('data/emb.npy', embedding)

