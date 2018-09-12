# coding: utf-8

# Cross Validation

import os
import gzip
import pickle
import numpy as np
import tensorflow as tf

from utils.utils import FeatureExtractor, LSTMCRFeeder, conll_format

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


train_set, valid_set, test_set, dicts = atisfold(1)
# train_set, test_set, dicts = atisfull()

w2idx, ne2idx, la2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']
idx2w = {w2idx[k]: k for k in w2idx}
idx2ne = {ne2idx[k]: k for k in ne2idx}
idx2la = {la2idx[k]: k for k in la2idx}

train_x, train_ne, train_la = train_set
val_x, val_ne, val_la = valid_set
test_x, test_ne, test_la = test_set


def dump_data(prefix, x, ne, la):
    wlength = 35
    with open('data/%s.data' % prefix, 'w') as fp:
        for sw, se, sl in zip(x, ne, la):
            for a, b, c in zip(sw, se, sl):
                fp.write(idx2w[a].ljust(wlength) + idx2ne[b].ljust(wlength) + idx2la[c].ljust(wlength) + '\n')
            fp.write('\n')


print('Dump data...')

dump_data('train', train_x, train_ne, train_la)
dump_data('valid', val_x, val_ne, val_la)
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
val_feats = fe.extract_features('data/valid.data')
test_feats = fe.extract_features('data/test.data')

print('Load model...')

num_classes = len(la2idx.keys())
max_length = max(
    max(map(len, train_la)),
    max(map(len, test_la)),
)
vocab_size = len(w2idx)

model = BiLSTMCRFModel(False, fe.feat_size, vocab_size, 128, 256, num_classes, max_length, 0.00001, 0.001, 1.0)

print('Start training...')

max_epoch = 50

# saver = tf.train.Saver()

# feeder = LSTMCRFeeder(train_x, train_feats, train_la, max_length, model.feat_size, 64)
# tokens, feats, labels = feeder.feed()

sess = tf.Session()

sess.run(tf.global_variables_initializer())
feeder = LSTMCRFeeder(train_x, train_feats, train_la, max_length, model.feat_size, 64)

emb = np.load('data/emb.npy')
model.init_embedding(sess, emb)

pre_f1 = []

for epoch in range(1, max_epoch + 1):
    loss = 0
    for step in range(feeder.step_per_epoch):
        tokens, feats, labels = feeder.feed()

        step_loss, l2 = model.train_step(sess, tokens, feats, labels)
        loss += step_loss

        print('epoch: %d, size: %d/%d, step_loss: %f, epoch_loss: %f' %
              (epoch, feeder.offset, feeder.size, step_loss, loss)
              )

    tokens, feats = feeder.test(val_x, val_feats)
    pred = model.test(sess, tokens, feats)
    f1 = conll_format(val_x, val_la, pred, idx2w, idx2la, 'valid')

    '''
    if len(pre_f1) < 5:
        pre_f1.append(f1)
    else:
        # Early Stop
        avg = sum(pre_f1) / len(pre_f1)
        if avg > f1:
            print("\nFind best epoch = %d, best F1 = %f" % (epoch, f1))
            break
        else:
            pre_f1.pop(0)
            pre_f1.append(f1)
            print("\nEpoch %d, F1 = %f" % (epoch, f1))
    '''
    pre_f1.append((epoch, f1))
    print("\nEpoch %d, F1 = %f" % (epoch, f1))

    # tokens, feats, length = feeder.predict(test_x[0], test_feats[0])
    # labels = test_la[0]
    # pred, scores = model.decode(sess, tokens, feats, length, 10)
    # print('{:<20} {}'.format('golden', labels.tolist()))

    # saver.save(sess, 'checkpoints/model.ckpt', global_step=model.global_step)

    print('')
    feeder.next_epoch()

'''
print('Predict...')
tokens, feats = feeder.test(test_x, test_feats)
pred = model.test(sess, tokens, feats)

print('Dump conll format...')
conll_format(test_x, test_la, pred, idx2w, idx2la, 'test')

compare = np.array(list(map(lambda zz: np.array_equal(zz[0], zz[1]), zip(test_la, pred))))
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
