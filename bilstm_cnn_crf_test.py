# coding: utf-8

import os
import pickle
import logging

from tqdm import tqdm
import tensorflow as tf

from utils.feeder.LSTMCNNCRFeeder import LSTMCNNCRFeeder
from utils.parser import parse_conll2003
from utils.checkmate import BestCheckpointSaver, best_checkpoint
from utils.conlleval import evaluate

from model.BiLSTMCNNCRF import BiLSTMCNNCRFModel


def conll2003():
    if not os.path.isfile('dev/conll.pkl'):
        parse_conll2003()
    with open('dev/conll.pkl', 'rb') as fp:
        train_set, val_set, test_set, dicts = pickle.load(fp)

    return train_set, val_set, test_set, dicts


train_set, val_set, test_set, dicts = conll2003()

w2idx, la2idx = dicts['words2idx'], dicts['labels2idx']
idx2w = {w2idx[k]: k for k in w2idx}
idx2la = {la2idx[k]: k for k in la2idx}

train_x, train_chars, train_la = train_set
val_x, val_chars, val_la = val_set
test_x, test_chars, test_la = test_set

print('Load model...')

num_classes = len(la2idx.keys())
max_seq_length = max(
    max(map(len, train_x)),
    max(map(len, test_x)),
)
max_word_length = max(
    max([len(ssc) for sc in train_chars for ssc in sc]),
    max([len(ssc) for sc in test_chars for ssc in sc])
)

model = BiLSTMCNNCRFModel(
    True,
    50,  # Word embedding size
    16,  # Character embedding size
    100,  # LSTM state size
    128,  # Filter num
    3,  # Filter size
    num_classes,
    max_seq_length,
    max_word_length,
    0.015,
    0.5)

print('Start training...')
print('Train size = %d' % len(train_x))
print('Val size = %d' % len(val_x))
print('Test size = %d' % len(test_x))
print('Num classes = %d' % num_classes)

start_epoch = 1
max_epoch = 1000

saver = tf.train.Saver()
best_saver = BestCheckpointSaver(
    save_dir='checkpoints/best',
    num_to_keep=1,
    maximize=True
)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.FileHandler('logs/train.log'), logging.StreamHandler()])

best_checkpoint = best_checkpoint('checkpoints/best/', True)
if best_checkpoint:
    saver.restore(sess, best_checkpoint)
else:
    sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

train_feeder = LSTMCNNCRFeeder(train_x, train_chars, train_la, max_seq_length, max_word_length, 16)
val_feeder = LSTMCNNCRFeeder(val_x, val_chars, val_la, max_seq_length, max_word_length, 16)
test_feeder = LSTMCNNCRFeeder(test_x, test_chars, test_la, max_seq_length, max_word_length, 16)

preds = []
for step in tqdm(range(test_feeder.step_per_epoch)):
    tokens, chars, labels = test_feeder.feed()
    pred = model.test(sess, tokens, chars)
    preds.extend(pred)
true_seqs = [idx2la[la] for sl in test_la for la in sl]
pred_seqs = [idx2la[la] for sl in preds for la in sl]
ll = min(len(true_seqs), len(pred_seqs))
_, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

test_feeder.next_epoch(False)

print("Test F1: %f" % f1)

'''
total = len(test_la)
with open('dev/test.format', 'w') as fp:
    for step in tqdm(range(test_feeder.step_per_epoch)):
        tokens, chars, labels = test_feeder.feed()
        pred = model.test(sess, tokens, chars)

        true_seqs = [[idx2la[la] for la in sl] for sl in labels]
        pred_seqs = [[idx2la[la] for la in sl] for sl in pred]

        for st, sl, sp in zip(tokens, true_seqs, pred_seqs):
            for tup in zip(st, sl, sp):
                fp.write(' '.join(tup) + '\n')
            fp.write('\n')
'''


def dump_topK(prefix, feeder, topK):
    total = len(test_la)

    with open('dev/predict.%s' % prefix, 'w') as fp:
        for i in range(total):
            tokens, chars, labels = feeder.feed()
            pred = model.test(sess, tokens, chars)

            true_seqs = [[idx2la[la] for la in sl] for sl in labels]
            pred_seqs = [[idx2la[la] for la in sl] for sl in pred]

            for st, sl, sp in zip(tokens, true_seqs, pred_seqs):
                for tup in zip(st, sl, sp):
                    fp.write(' '.join(tup) + '\n')
                fp.write('\n')


# train_feeder = LSTMCNNCRFeeder(train_x, train_chars, train_la, max_seq_length, max_word_length, 16)

dump_topK('test', test_feeder, 3)
