# coding: utf-8

import os
import gzip
import pickle
import numpy as np
import tensorflow as tf

from utils.utils import FeatureExtractor, LSTMCRFeeder, LSTMCNNCRFeeder, conll_format

from model.BiLSTMCNNCRF import BiLSTMCNNCRFModel


def conll2003():
    with open('data/conll.pkl', 'rb') as fp:
        train_set, val_set, test_set, dicts = pickle.load(fp)
    return train_set, val_set, test_set, dicts


train_set, val_set, test_set, dicts = conll2003()

w2idx, ch2idx, la2idx = dicts['words2idx'], dicts['chars2idx'], dicts['labels2idx']
idx2w = {w2idx[k]: k for k in w2idx}
idx2ch = {ch2idx[k]: k for k in ch2idx}
idx2la = {la2idx[k]: k for k in la2idx}

train_x, train_chars, train_la = train_set
val_x, val_chars, val_la = val_set
test_x, test_chars, test_la = test_set


def dump_data(prefix, x, la):
    wlength = 50
    with open('data/%s.data' % prefix, 'w') as fp:
        for sw, sl in zip(x, la):
            for a, b in zip(sw, sl):
                fp.write(idx2w[a].ljust(wlength) + idx2la[b].ljust(wlength) + '\n')
            fp.write('\n')


print('Dump data...')

dump_data('train', train_x, train_la)
dump_data('val', val_x, val_la)
dump_data('test', test_x, test_la)

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
max_seq_length = max(
    max(map(len, train_x)),
    max(map(len, test_x)),
)
max_char_length = max(
    max([len(ssc) for sc in train_chars for ssc in sc]),
    max([len(ssc) for sc in test_chars for ssc in sc])
)
vocab_size = len(w2idx)
char_size = len(ch2idx)


# model = BiLSTMCRFModel(True, fe.feat_size, vocab_size, 128, 256, num_classes, max_length, 0.00001, 0.001, 1.0)
model = BiLSTMCNNCRFModel(
    True,
    fe.feat_size,
    vocab_size,
    char_size,
    128,
    32,
    256,
    num_classes,
    max_seq_length,
    max_char_length,
    0.001,
    0.5)

print('Start training...')

max_epoch = 10

saver = tf.train.Saver()

# feeder = LSTMCRFeeder(train_x, train_feats, train_la, max_length, model.feat_size, 32)
# tokens, feats, labels = feeder.feed()

sess = tf.Session()

sess.run(tf.global_variables_initializer())
feeder = LSTMCNNCRFeeder(train_x,
                      train_chars,
                      train_feats,
                      train_la,
                      max_seq_length,
                      max_char_length,
                      model.feat_size,
                      16)

# emb = np.load('data/emb.npy')
# model.init_embedding(sess, emb)

for epoch in range(1, max_epoch + 1):
    loss = 0
    for step in range(feeder.step_per_epoch):
        tokens, chars, feats, labels = feeder.feed()

        step_loss = model.train_step(sess, tokens, chars, feats, labels)
        loss += step_loss

        print('epoch: %d, size: %d/%d, step_loss: %f, epoch_loss: %f' %
              (epoch, feeder.offset, feeder.size, step_loss, loss)
        )

    tokens, chars, feats, length = feeder.predict(test_x[0], test_chars[0], test_feats[0])
    labels = test_la[0]
    pred, scores = model.decode(sess, tokens, feats, length, 10)
    print('{:<20} {}'.format('golden', labels.tolist()))
    # print(pred)

    saver.save(sess, 'checkpoints/model.ckpt', global_step=model.global_step)

    print('')
    feeder.next_epoch()

print('Predict...')
tokens, chars, feats = feeder.test(test_x, test_chars, test_feats)
pred = model.test(sess, tokens, chars, feats)

print('Dump conll format...')
conll_format(test_x, test_la, pred, idx2w, idx2la, 'test')

compare = np.array(list(map(lambda zz : np.array_equal(zz[0], zz[1]), zip(test_la, pred))))
error_idx = np.where(compare == False)[0]


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


'''
def dump_topK(topK):
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

def dump_topK(topK):
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
'''
