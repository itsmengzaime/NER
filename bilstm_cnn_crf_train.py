# coding: utf-8

import os
import pickle
import logging

import tensorflow as tf

from utils.feeder.LSTMCNNCRFeeder import LSTMCNNCRFeeder
from utils.parser import parse_conll2003
from utils.conlleval import evaluate
from utils.checkmate import BestCheckpointSaver, best_checkpoint

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
    50,   # Word embedding size
    16,   # Character embedding size
    100,  # LSTM state size
    128,  # Filter num
    3,    # Filter size
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

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir='checkpoints')
if latest_checkpoint:
    saver.restore(sess, latest_checkpoint)
else:
    sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# feeder = LSTMCNNCRFeeder(train_x, train_chars, train_la, max_seq_length, max_word_length, 10)
# tokens, chars, labels = feeder.feed()
# a, b = sess.run([model.length, model.word_string_tensor], {model.tokens: tokens, model.chars: chars})

train_feeder = LSTMCNNCRFeeder(train_x, train_chars, train_la, max_seq_length, max_word_length, 16)
val_feeder = LSTMCNNCRFeeder(val_x, val_chars, val_la, max_seq_length, max_word_length, 16)
test_feeder = LSTMCNNCRFeeder(test_x, test_chars, test_la, max_seq_length, max_word_length, 16)

for epoch in range(start_epoch, max_epoch + 1):
    loss = 0
    for step in range(train_feeder.step_per_epoch):
        tokens, chars, labels = train_feeder.feed()

        step_loss = model.train_step(sess, tokens, chars, labels)
        loss += step_loss

        logging.info('Epoch: %d, size: %d/%d, step_loss: %f, epoch_loss: %f' %
                     (epoch, train_feeder.offset, train_feeder.size, step_loss, loss)
                     )

    preds = []
    for step in range(val_feeder.step_per_epoch):
        tokens, chars, labels = val_feeder.feed()
        pred = model.test(sess, tokens, chars)
        preds.extend(pred)
    true_seqs = [idx2la[la] for sl in val_la for la in sl]
    pred_seqs = [idx2la[la] for sl in preds for la in sl]
    ll = min(len(true_seqs), len(pred_seqs))
    _, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

    val_feeder.next_epoch(False)

    logging.info("Epoch: %d, val_f1: %f" % (epoch, f1))

    preds = []
    for step in range(test_feeder.step_per_epoch):
        tokens, chars, labels = test_feeder.feed()
        pred = model.test(sess, tokens, chars)
        preds.extend(pred)
    true_seqs = [idx2la[la] for sl in test_la for la in sl]
    pred_seqs = [idx2la[la] for sl in preds for la in sl]
    ll = min(len(true_seqs), len(pred_seqs))
    _, _, f1 = evaluate(true_seqs[:ll], pred_seqs[:ll], False)

    test_feeder.next_epoch(False)

    logging.info("Epoch: %d, test_f1: %f" % (epoch, f1))

    saver.save(sess, 'checkpoints/model.ckpt', global_step=epoch)
    best_saver.handle(f1, sess, epoch)

    logging.info('')
    train_feeder.next_epoch()
