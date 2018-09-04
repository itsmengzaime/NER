# coding: utf-8

import os
import gzip
import pickle
import tensorflow as tf

from utils.utils import FeatureExtractor, CRFeeder, conll_format

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

w2idx, ne2idx, la2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']
idx2w = {w2idx[k]: k for k in w2idx}
idx2ne = {ne2idx[k]: k for k in ne2idx}
idx2la = {la2idx[k]: k for k in la2idx}

train_x, train_ne, train_la = train_set
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

model = CRFModel(fe.feat_size, num_classes, max_length, 0.00001, 0.001)

print('Start training...')

max_epoch = 50

saver = tf.train.Saver()

# feeder = Feeder(train_feats, train_la, max_length, model.feat_size, 64)
# X, y = feeder.feed()

sess = tf.Session()

sess.run(tf.global_variables_initializer())
feeder = CRFeeder(train_feats, train_la, max_length, model.feat_size, 64)

for epoch in range(1, max_epoch + 1):
    loss = 0
    for step in range(feeder.step_per_epoch):
        X, y = feeder.feed()

        step_loss, l2 = model.train_step(sess, X, y)
        loss += step_loss

        print('epoch: %d, size: %d/%d, step_loss: %f, epoch_loss: %f' %
              (epoch, feeder.offset, feeder.size, step_loss, loss)
        )

    test_X, length = feeder.predict(test_feats[0])
    test_y = test_la[0]
    pred = model.decode(sess, test_X, length)
    print(test_y.tolist())
    print(pred)

    saver.save(sess, 'checkpoints/model.ckpt', global_step=model.global_step)

    print('')
    feeder.next_epoch()

print('Predict...')
X = feeder.test(test_feats)
pred = model.test(sess, X)

print('Dump conll format...')
conll_format(test_x, test_la, pred, idx2w, idx2la, 'test')
