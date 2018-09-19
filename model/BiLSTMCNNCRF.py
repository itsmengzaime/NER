# encoding: utf-8

from math import sqrt

import numpy as np
import tensorflow as tf

from utils.utils import viterbi_decode_topk


class BiLSTMCNNCRFModel(object):
    """Bi-LSTM + CRF implemented by Tensorflow

    Attributes:
        num_classes: number of classes
        max_length: max length of sentence
        learning_rate: learning rate
    """

    def __init__(self, pre_embedding: bool,
                 word_embed_size: int,
                 char_embed_size: int,
                 hidden_size: int,
                 filter_size: int,
                 num_classes: int,
                 max_seq_length: int,
                 max_word_length: int,
                 learning_rate: float,
                 dropout: float):
        self.pre_embedding = pre_embedding
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length
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
        word_vocab = []
        with open('dev/train.word.vocab') as fp:
            for row in fp:
                word_vocab.append(row.strip())

        char_vocab = []
        with open('dev/train.char.vocab') as fp:
            for row in fp:
                word_vocab.append(row.strip())

        return word_vocab, char_vocab

    def _add_placeholders(self):
        self.tokens = tf.placeholder(tf.string, [None, self.max_seq_length])
        self.chars = tf.placeholder(tf.string, [None, self.max_seq_length, self.max_word_length])
        self.dropout = tf.placeholder(tf.float32)
        self.labels = tf.placeholder(tf.int32, [None, self.max_seq_length])
        self.length = tf.count_nonzero(self.tokens, axis=1)

    def _add_embedding(self):
        with tf.variable_scope('embedding'):
            train_word_vocab, train_char_vocab = self._load_train_vocab()
            if self.pre_embedding:
                pretrained_vocab, pretrained_embs = self._load_pretrained_senna()

                only_in_train = list(set(train_word_vocab) - set(pretrained_vocab))
                vocab = pretrained_vocab + only_in_train

                vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(vocab),
                    default_value=len(vocab)
                )
                word_string_tensor = vocab_lookup.lookup(self.tokens)

                pretrained_embs = tf.get_variable(
                    name='embs_pretrained',
                    initializer=tf.constant_initializer(np.asarray(pretrained_embs), dtype=tf.float32),
                    shape=pretrained_embs.shape,
                    trainable=False
                )
                train_embs = tf.get_variable(
                    name='embs_only_in_train',
                    shape=[len(only_in_train), self.word_embed_size],
                    initializer=tf.random_uniform_initializer(-sqrt(3 / self.word_embed_size),
                                                              sqrt(3 / self.word_embed_size)),
                    trainable=True
                )
                unk_embs = tf.get_variable(
                    name='embs_unk',
                    shape=[1, self.word_embed_size],
                    initializer=tf.random_uniform_initializer(-sqrt(3 / self.word_embed_size),
                                                              sqrt(3 / self.word_embed_size)),
                    trainable=False
                )
                word_embeddings = tf.concat([pretrained_embs, train_embs, unk_embs], axis=0)
            else:
                word_embeddings = tf.get_variable(
                    name='embeds_word',
                    shape=[len(train_word_vocab) + 1, self.word_embed_size])
                vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(train_word_vocab),
                    default_value=len(train_word_vocab)
                )
                word_string_tensor = vocab_lookup.lookup(self.tokens)

            # word_embeddings = tf.nn.dropout(word_embeddings, keep_prob=self.dropout)
            self.word_embedding_layer = tf.nn.embedding_lookup(word_embeddings, word_string_tensor)

            char_embeddings = tf.get_variable(
                name='embs_char',
                shape=[len(train_char_vocab) + 1, self.char_embed_size],
                initializer=tf.random_uniform_initializer(-sqrt(3 / self.char_embed_size),
                                                          sqrt(3 / self.char_embed_size)),
                trainable=True
            )
            vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(train_char_vocab),
                default_value=len(train_char_vocab)
            )
            char_string_tensor = vocab_lookup.lookup(self.chars)

            # char_embeddings = tf.nn.dropout(char_embeddings, keep_prob=self.dropout)
            self.char_embedding_layer = tf.nn.embedding_lookup(char_embeddings, char_string_tensor)

    def _add_cnn(self):
        with tf.variable_scope('cnn'):
            # char_embed: [batch_size, max_seq_length, max_char_length, char_embed_size]
            flat = tf.reshape(self.char_embedding_layer, (-1, self.max_word_length, self.char_embed_size))
            conv = tf.layers.conv1d(flat, filters=self.filter_size, kernel_size=3,
                                    padding='same')  # [batch_size, max_seq_length, max_char_length, 30]
            conv = tf.reshape(conv, (-1, self.max_seq_length, self.max_word_length, self.char_embed_size))
            pool = tf.reduce_max(conv, axis=2)  # [batch_size, max_seq_length, 30]

        self.embedding_layer = tf.concat([self.word_embedding_layer, pool], axis=2)
        self.embedding_layer = tf.nn.dropout(self.embedding_layer, keep_prob=self.dropout)

    def _add_rnn(self):
        def rnn_cell(gru=True):
            if gru:
                cell = tf.contrib.rnn.GRUCell(self.hidden_size,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())
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
        with tf.variable_scope('linear'):
            w = tf.get_variable('w', [self.hidden_size * 2, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [self.num_classes], initializer=tf.contrib.layers.xavier_initializer())

        flattened_potentials = tf.matmul(flattened_output, w) + b
        self.unary_potentials = tf.reshape(
            flattened_potentials,
            [-1, self.max_seq_length, self.num_classes]
        )
        self.ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(
            self.unary_potentials, self.labels, self.length
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

    def train_step(self, sess, tokens, chars, labels):
        input_feed = {
            self.tokens: tokens,
            self.chars: chars,
            self.labels: labels,
            self.dropout: self.dropout
        }
        output_feed = [
            self._train_op,
            self.loss
        ]

        _, loss = sess.run(output_feed, input_feed)
        return loss

    def test(self, sess, tokens, chars):
        viterbi_sequences, lengths = sess.run(
            [self.viterbi_sequence, self.length], {
                self.tokens: tokens,
                self.chars: chars,
                self.dropout: 1.0
            }
        )

        pred = []
        for i in range(lengths.shape[0]):
            length = lengths[i]
            pred.append(viterbi_sequences[i, :length])

        return pred

    def decode(self, sess, tokens, chars, length, topK=5):
        '''
        score: [seq_len, num_tags]
        transition_params: [num_tags, num_tags]
        '''

        score, trans_params = sess.run([self.unary_potentials, self.trans_params], {
            self.tokens: tokens,
            self.chars: chars,
            self.dropout: 1.0
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
