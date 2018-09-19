# encoding: utf-8

import numpy as np
import tensorflow as tf

from utils.utils import viterbi_decode_topk


class BiLSTMCRFModel(object):
    """Bi-LSTM + CRF implemented by Tensorflow

    Attributes:
        pre_embedding: use pre embedding
        feat_size: feature size
        vocab_size: vocabulary size
        embed_size: word embedding size
        hidden_size: rnn hidden size
        num_classes: number of classes
        max_length: max length of sentence
        learning_rate: learning rate
        dropout: keep probability for dropout layer
    """

    def __init__(self,
                 pre_embedding: bool,
                 feat_size: int,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int,
                 num_classes: int,
                 max_length: int,
                 learning_rate: float,
                 dropout: float):
        self.pre_embedding = pre_embedding
        self.feat_size = feat_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.dropout = dropout

        self._build_graph()

    def _load_pretrained_senna(self):
        vocab = []
        with open('senna/pretrained.vocab') as fp:
            for row in fp:
                vocab.append(row.strip())
        emb = np.genfromtxt('senna/pretrained.emb', delimiter=' ', dtype=np.float)

        return vocab, emb

    def _load_train_vocab(self):
        vocab = []
        with open('dev/train.vocab') as fp:
            for row in fp:
                vocab.append(row.strip())

        return vocab

    def _add_placeholders(self):
        self._tokens = tf.placeholder(tf.string, [None, self.max_length])
        self._dropout = tf.placeholder(tf.float32)
        self._feats = tf.sparse_placeholder(
            tf.float32,
            [None, self.max_length, self.feat_size]
        )
        self.dense_feats = tf.sparse_tensor_to_dense(self._feats, validate_indices=False)
        self._labels = tf.placeholder(tf.int32, [None, self.max_length])
        self.mask = tf.sign(tf.reduce_sum(self.dense_feats, axis=2))
        self.length = tf.cast(tf.reduce_sum(self.mask, axis=1), tf.int32)

    def _add_embedding(self):
        with tf.variable_scope('embedding'):
            if self.pre_embedding:
                pretrained_vocab, pretrained_embs = self._load_pretrained_senna()
                train_vocab = self._load_train_vocab()
                only_in_train = list(set(train_vocab) - set(pretrained_vocab))
                vocab = pretrained_vocab + only_in_train

                vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(vocab),
                    default_value=len(vocab)
                )
                string_tensor = vocab_lookup.lookup(self._tokens)

                pretrained_embs = tf.get_variable(
                    name='embs_pretrained',
                    initializer=tf.constant_initializer(np.asarray(pretrained_embs), dtype=tf.float32),
                    shape=pretrained_embs.shape,
                    trainable=False
                )
                train_embs = tf.get_variable(
                    name='embs_only_in_train',
                    shape=[len(only_in_train), self.embed_size],
                    initializer=tf.random_uniform_initializer(-0.04, 0.04),
                    trainable=True
                )
                unk_embs = tf.get_variable(
                    name='embs_unk',
                    shape=[1, self.embed_size],
                    initializer=tf.random_uniform_initializer(-0.04, 0.04),
                    trainable=False
                )

                embeddings = tf.concat([pretrained_embs, train_embs, unk_embs], axis=0)
            else:
                embeddings = tf.get_variable('embed', [self.vocab_size, self.embed_size])
            embeddings = tf.nn.dropout(embeddings, keep_prob=self._dropout)
            self.embedding_layer = tf.nn.embedding_lookup(embeddings, string_tensor)

    def _add_rnn(self):

        def rnn_cell(gru=True):
            if gru:
                cell = tf.contrib.rnn.GRUCell(self.hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._dropout)
            return cell

        with tf.variable_scope('recurrent'):
            fw_cells = [rnn_cell(False) for _ in range(1)]
            bw_cells = [rnn_cell(False) for _ in range(1)]
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells,
                self.embedding_layer,
                dtype=tf.float32,
                sequence_length=tf.cast(self.length, tf.int64)
            )
            self.layer_output = tf.concat(axis=2, values=outputs)

    def _add_crf(self):
        flattened_output = tf.reshape(self.layer_output, [-1, self.hidden_size * 2])
        # flattened_feats = tf.reshape(self.dense_feats, [-1, self.feat_size])
        with tf.variable_scope('linear'):
            w1 = tf.get_variable('w1', [self.hidden_size * 2, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable('w2', [self.feat_size, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [self.num_classes], initializer=tf.contrib.layers.xavier_initializer())

        # flattened_potentials = tf.matmul(flattened_output, w1) + tf.matmul(flattened_feats, w2) + b
        flattened_potentials = tf.matmul(flattened_output, w1) + b
        self.unary_potentials = tf.reshape(
            flattened_potentials,
            [-1, self.max_length, self.num_classes]
        )
        self.ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(
            self.unary_potentials, self._labels, self.length
        )
        self.viterbi_sequence, _ = tf.contrib.crf.crf_decode(
            self.unary_potentials, self.trans_params, self.length
        )
        self.loss = tf.reduce_sum(-self.ll)
        tf.summary.scalar('loss', self.loss)

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
        self._add_rnn()
        self._add_crf()
        self._add_train_op()

        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def train_step(self, sess, tokens, feats, labels):
        input_feed = {
            self._tokens: tokens,
            self._feats: feats,
            self._labels: labels,
            self._dropout: self.dropout
        }
        output_feed = [
            self._train_op,
            self.loss,
        ]

        _, loss = sess.run(output_feed, input_feed)
        return loss

    def test(self, sess, tokens, feats):
        viterbi_sequences, lengths = sess.run(
            [self.viterbi_sequence, self.length], {
                self._tokens: tokens,
                self._feats: feats,
                self._dropout: 1.0
            }
        )

        pred = []
        for i in range(lengths.shape[0]):
            length = lengths[i]
            pred.append(viterbi_sequences[i, :length])

        return pred

    def decode(self, sess, tokens, feats, length, topK = 5):
        '''
        score: [seq_len, num_tags]
        transition_params: [num_tags, num_tags]
        '''

        score, trans_params = sess.run([self.unary_potentials, self.trans_params], {
            self._tokens: tokens,
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