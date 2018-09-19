# coding: utf-8

import os
import gzip
import pickle
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from utils.feeder.LSTMCRFeeder import LSTMCRFeeder
from utils.feature_extractor import FeatureExtractor
from utils.utils import conll_format
from utils.parser import parse_conll2003
from utils.conlleval import evaluate

from model.BiLSTMCRF import BiLSTMCRFModel


def atisfold(fold):
    assert fold in range(5)

    fp = gzip.open('data/atis.fold' + str(fold) + '.pkl.gz')
    train_set, valid_set, test_set, dicts = pickle.load(fp, encoding='iso-8859-1')
    return train_set, valid_set, test_set, dicts


def atisfull():
    with open('data/atis.pkl', 'rb') as fp:
        train_set, test_set, dicts = pickle.load(fp, encoding='iso-8859-1')
    return train_set, test_set, dicts


def conll2003():
    with open('dev/conll.pkl', 'rb') as fp:
        train_set, val_set, test_set, dicts = pickle.load(fp)
    return train_set, val_set, test_set, dicts


train_set, val_set, test_set, dicts = conll2003()

w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
idx2w = {w2idx[k]: k for k in w2idx}
idx2la = {la2idx[k]: k for k in la2idx}

train_x, _, train_la = train_set
val_x, _, val_la = val_set
test_x, _, test_la = test_set


print('Load data...')

fe = FeatureExtractor()
fe.parse_template('data/template')

template_vocab_dir = 'dev/template.vocabs'
if os.path.exists(template_vocab_dir):
    fe.construct_vocabs_from_file(template_vocab_dir)
else:
    os.mkdir(template_vocab_dir)
    fe.construct_vocabs_from_data('data/train.txt')
    fe.save_vocabs(template_vocab_dir)

'''
[train_size, max_length, feat_size]
feat_size: All unigram features
'''
train_feats = fe.extract_features('data/train.txt')
val_feats = fe.extract_features('data/valid.txt')
test_feats = fe.extract_features('data/test.txt')

print('Load model...')

num_classes = len(la2idx.keys())
max_length = max(
    max(map(len, train_la)),
    max(map(len, test_la)),
)
vocab_size = len(w2idx)

model = BiLSTMCRFModel(True, fe.feat_size, vocab_size, 50, 256, num_classes, max_length, 0.001, 0.5)

print('Start testing...')

saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir='checkpoints'))
sess.run(tf.tables_initializer())


val_feeder = LSTMCRFeeder(val_x, val_feats, val_la, max_length, model.feat_size, 16)
test_feeder = LSTMCRFeeder(test_x, test_feats, test_la, max_length, model.feat_size, 16)

preds = []
for step in tqdm(range(val_feeder.step_per_epoch)):
    tokens, feats, labels = val_feeder.feed()
    pred = model.test(sess, tokens, feats)
    preds.extend(pred)
true_seqs = [idx2la[la] for sl in test_la for la in sl]
pred_seqs = [idx2la[la] for sl in preds for la in sl]
ll = min(len(true_seqs), len(pred_seqs))
_, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

print('Val F1 = %f' % f1)

preds = []
for step in tqdm(range(test_feeder.step_per_epoch)):
    tokens, feats, labels = test_feeder.feed()
    pred = model.test(sess, tokens, feats)
    preds.extend(pred)
true_seqs = [idx2la[la] for sl in test_la for la in sl]
pred_seqs = [idx2la[la] for sl in preds for la in sl]
ll = min(len(true_seqs), len(pred_seqs))
_, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

print('Test F1 = %f' % f1)

'''
print('Dump conll format...')
conll_format(test_x, test_la, pred, idx2w, idx2la, 'test')

compare = np.array(list(map(lambda zz : np.array_equal(zz[0], zz[1]), zip(test_la, pred))))
error_idx = np.where(compare == False)[0]
'''


def eval(idx):
    tokens, feats, length = feeder.predict(test_x[idx], test_feats[idx])
    model.decode(sess, tokens, feats, length)
    print('{:<20} {}'.format('golden', test_la[idx].tolist()))


def count_in(topK):
    total = len(error_idx)
    total_in = 0
    for idx in error_idx:
        tokens, feats, length = feeder.predict(test_x[idx], test_feats[idx])
        pred, scores = model.decode(sess, tokens, feats, length, topK)

        golden_la = test_la[idx].tolist()

        if golden_la in pred:
            total_in += 1

    print('{}/{} {}'.format(total_in, total, total_in / total))


def dump_train_topK(topK):
    total = len(train_la)

    with open('dev/train.format', 'w') as fp:
        for i in range(total):
            list_x = train_x[i]
            list_feats = train_feats[i]
            list_label = train_la[i]

            tokens, feats, length = feeder.predict(list_x, list_feats)
            preds, scores = model.decode(sess, tokens, feats, length, topK)

            for j in range(length):
                out = []
                out.append(idx2w[list_x[j]])
                out.append(idx2la[list_label[j]])
                for k in range(topK):
                    out.append(idx2la[preds[k][j]])
                fp.write('\t'.join(out))
                fp.write('\n')
            fp.write('\n')


def dump_test_topK(topK):
    total = len(test_la)

    with open('dev/test.format', 'w') as fp:
        for i in range(total):
            list_x = test_x[i]
            list_feats = test_feats[i]
            list_label = test_la[i]

            tokens, feats, length = feeder.predict(list_x, list_feats)
            preds, scores = model.decode(sess, tokens, feats, length, topK)

            for j in range(length):
                out = []
                out.append(idx2w[list_x[j]])
                out.append(idx2la[list_label[j]])
                for k in range(topK):
                    out.append(idx2la[preds[k][j]])
                fp.write('\t'.join(out))
                fp.write('\n')
            fp.write('\n')


# dump_test_topK(1)
