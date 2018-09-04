# encoding: utf-8

import numpy as np
import tensorflow as tf


class CRFModel(object):
    """CRF Model implemented by Tensorflow

    Attributes:
        feat_size: feature size
        num_classes: number of classes
        max_length: max length of sentence
        weight_decay: weight of l2 loss
        learning_rate: learning rate
    """

    def __init__(self, feat_size: int, num_classes: int, max_length: int, weight_decay: float, learning_rate: float):
        self.feat_size = feat_size
        self.num_classes = num_classes
        self.max_length = max_length
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self._build_graph()

    def _add_placeholders(self):
        self._feats = tf.sparse_placeholder(
            tf.float32,
            [None, self.max_length, self.feat_size]
        )
        self.dense_feats = tf.sparse_tensor_to_dense(self._feats, validate_indices=False)
        self._labels = tf.placeholder(tf.int32, [None, self.max_length])
        self.mask = tf.sign(tf.reduce_sum(self.dense_feats, axis=2))
        self.length = tf.cast(tf.reduce_sum(self.mask, axis=1), tf.int32)

    def _add_crf(self):
        flattened_feats = tf.reshape(self.dense_feats, [-1, self.feat_size])
        with tf.variable_scope('linear'):
            w = tf.get_variable('w', [self.feat_size, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
        flattened_potentials = tf.matmul(flattened_feats, w) + b
        self.unary_potentials = tf.reshape(
            flattened_potentials,
            [-1, self.max_length, self.num_classes]
        )
        self.ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(
            self.unary_potentials, self._labels, self.length
        )
        self.loss = tf.reduce_sum(-self.ll)
        self.l2 = self.weight_decay * tf.reduce_sum(tf.nn.l2_loss(w))

    def _add_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        params = tf.trainable_variables()
        grads = tf.gradients(self.loss + self.l2, params)
        self._train_op = optimizer.apply_gradients(
            zip(grads, params),
            global_step=self.global_step
        )

    def _build_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self._add_placeholders()
        self._add_crf()
        self._add_train_op()

        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def train_step(self, sess, X, y):
        input_feed = {
            self._feats: X,
            self._labels: y
        }
        output_feed = [
            self._train_op,
            self.loss,
            self.l2
        ]

        _, loss, l2 = sess.run(output_feed, input_feed)
        return loss, l2

    def test(self, sess, X):
        score, trans_params, length = sess.run([self.unary_potentials, self.trans_params, self.length], {self._feats: X})

        pred = []
        for i in range(score.shape[0]):
            viterbi, viterbi_score = tf.contrib.crf.viterbi_decode(
                score[i, :],
                trans_params
            )
            pred.append(viterbi)

        return pred

    def decode(self, sess, X, length):
        '''
        score: [seq_len, num_tags]
        transition_params: [num_tags, num_tags]
        '''

        score, trans_params = sess.run([self.unary_potentials, self.trans_params], {self._feats: X})
        score = np.squeeze(score, 0)
        score = score[:length, :]

        viterbi, viterbi_score = tf.contrib.crf.viterbi_decode(
            score,
            trans_params
        )

        return viterbi