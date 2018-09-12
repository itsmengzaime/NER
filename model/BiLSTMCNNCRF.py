# encoding: utf-8

import numpy as np
import tensorflow as tf

from utils.utils import viterbi_decode_topk


class BiLSTMCNNCRFModel(object):
    """Bi-LSTM + CRF implemented by Tensorflow

    Attributes:
        feat_size: feature size
        num_classes: number of classes
        max_length: max length of sentence
        weight_decay: weight of l2 loss
        learning_rate: learning rate
    """

    def __init__(self, pre_embedding: bool,
                 feat_size: int,
                 vocab_size: int,
                 char_size: int,
                 word_embed_size: int,
                 char_embed_size: int,
                 hidden_size: int,
                 num_classes: int,
                 max_seq_length: int,
                 max_char_length: int,
                 learning_rate: float,
                 dropout: float):
        self.pre_embedding = pre_embedding
        self.feat_size = feat_size
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.max_char_length = max_char_length
        self.learning_rate = learning_rate
        self.dropout = dropout

        self._build_graph()

    def _add_placeholders(self):
        self._tokens = tf.placeholder(tf.int32, [None, self.max_seq_length])
        self._chars = tf.placeholder(tf.int32, [None, self.max_seq_length, self.max_char_length])
        self._dropout = tf.placeholder(tf.float32)
        self._feats = tf.sparse_placeholder(
            tf.float32,
            [None, self.max_seq_length, self.feat_size]
        )
        self._dense_feats = tf.sparse_tensor_to_dense(self._feats, validate_indices=False)
        self._labels = tf.placeholder(tf.int32, [None, self.max_seq_length])
        self._mask = tf.sign(tf.reduce_sum(self._dense_feats, axis=2))
        self.length = tf.cast(tf.reduce_sum(self._mask, axis=1), tf.int32)

    def init_embedding(self, sess, emb):
        if self.pre_embedding:
            sess.run(self._embedding_init_op, {self._embedding_placeholder: emb})

    def _add_embedding(self):
        with tf.variable_scope('embedding'):
            self.char_embed = tf.get_variable("char_embed", [self.char_size, self.char_embed_size])
            self.char_embed = tf.nn.embedding_lookup(self.char_embed, self._chars)
            if self.pre_embedding:
                word_embed = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.word_embed_size]), trainable=True, name="word_embed")
                self._embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.word_embed_size])
                self._embedding_init_op = word_embed.assign(self._embedding_placeholder)
            else:
                word_embed = tf.get_variable('word_embed', [self.vocab_size, self.word_embed_size])
            self.word_embed = tf.nn.embedding_lookup(word_embed, self._tokens)

    def _add_cnn(self):
        with tf.variable_scope('cnn'):
            # char_embed: [batch_size, max_seq_length, max_char_length, char_embed_size]
            flat = tf.reshape(self.char_embed, (-1, self.max_char_length, self.char_embed_size))
            conv = tf.layers.conv1d(flat, filters=32, kernel_size=3, padding='same')  # [batch_size, max_seq_length, max_char_length, 30]
            conv = tf.reshape(conv, (-1, self.max_seq_length, self.max_char_length, self.char_embed_size))
            pool = tf.reduce_max(conv, axis=2)                                                   # [batch_size, max_seq_length, 30]

        self.embedding_layer = tf.concat([self.word_embed, pool], axis=2)
        self.embedding_layer = tf.nn.dropout(self.embedding_layer,keep_prob=self.dropout)

    def _add_rnn(self):
        def rnn_cell(gru=True):
            if gru:
                cell = tf.contrib.rnn.GRUCell(self.hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._dropout)
            return cell

        with tf.variable_scope('recurrent'):
            fw_cells = [rnn_cell() for _ in range(2)]
            bw_cells = [rnn_cell() for _ in range(2)]
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells,
                self.embedding_layer,
                dtype=tf.float32,
                sequence_length=tf.cast(self.length, tf.int64)
            )
            self.layer_output = tf.concat(axis=2, values=outputs)

    def _add_crf(self):
        flattened_output = tf.reshape(self.layer_output, [-1, self.hidden_size * 2])
        flattened_feats = tf.reshape(self._dense_feats, [-1, self.feat_size])
        with tf.variable_scope('linear'):
            w1 = tf.get_variable('w1', [self.hidden_size * 2, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable('w2', [self.feat_size, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [self.num_classes], initializer=tf.contrib.layers.xavier_initializer())

        flattened_potentials = tf.matmul(flattened_output, w1) + tf.matmul(flattened_feats, w2) + b
        self.unary_potentials = tf.reshape(
            flattened_potentials,
            [-1, self.max_seq_length, self.num_classes]
        )
        self.ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(
            self.unary_potentials, self._labels, self.length
        )
        self.viterbi_sequence, _ = tf.contrib.crf.crf_decode(
            self.unary_potentials, self.trans_params, self.length
        )
        self.loss = tf.reduce_sum(-self.ll)

    def _add_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        self._train_op = optimizer.apply_gradients(
            zip(grads, params),
            global_step=self.global_step
        )

    def _build_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self._add_placeholders()
        self._add_embedding()
        self._add_cnn()
        self._add_rnn()
        self._add_crf()
        self._add_train_op()

        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def train_step(self, sess, tokens, chars, feats, labels):
        input_feed = {
            self._tokens: tokens,
            self._chars: chars,
            self._feats: feats,
            self._labels: labels,
            self._dropout: self.dropout
        }
        output_feed = [
            self._train_op,
            self.loss
        ]

        _, loss = sess.run(output_feed, input_feed)
        return loss

    def test(self, sess, tokens, chars, feats):
        viterbi_sequences, lengths = sess.run(
            [self.viterbi_sequence, self.length], {
                self._tokens: tokens,
                self._chars: chars,
                self._feats: feats,
                self._dropout: 1.0
            }
        )

        pred = []
        for i in range(lengths.shape[0]):
            length = lengths[i]
            pred.append(viterbi_sequences[i, :length])

        return pred

    def decode(self, sess, tokens, chars, feats, length, topK = 5):
        '''
        score: [seq_len, num_tags]
        transition_params: [num_tags, num_tags]
        '''

        score, trans_params = sess.run([self.unary_potentials, self.trans_params], {
            self._tokens: tokens,
            self._chars: chars,
            self._feats: feats,
            self._dropout: 1.0
        })
        score = np.squeeze(score, 0)
        score = score[:length, :]

        '''
        viterbi, viterbi_score = tf.contrib.crf.viterbi_decode(
            score,
            trans_params
        )
        print("{:<20} {}".format(viterbi_score, viterbi))
        '''

        viterbi, viterbi_score = viterbi_decode_topk(
            score,
            trans_params,
            topK
        )

        for a, b in zip(viterbi_score, viterbi):
            print("{:<20} {}".format(a, b))

        return viterbi, viterbi_score