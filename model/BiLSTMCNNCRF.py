# encoding: utf-8

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

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
        self._dropout = dropout

        self._build_graph()

    def _load_pretrained_senna(self):
        vocab = []
        with open('senna/pretrained.vocab') as fp:
            for row in fp:
                vocab.append(row.strip())
        emb = np.genfromtxt('senna/pretrained.emb', delimiter=' ', dtype=np.float)

        return vocab, emb

    def _load_pretrained_glove(self):
        model = KeyedVectors.load_word2vec_format('glove/glove.6B.100d.txt')

        vocab = list(model.vocab.keys())
        emb = model.vectors

        return vocab, emb

    def _load_train_vocab(self):
        word_vocab = []
        with open('dev/train.word.vocab') as fp:
            for row in fp:
                word_vocab.append(row.strip())

        char_vocab = []
        with open('dev/train.char.vocab') as fp:
            for row in fp:
                char_vocab.append(row.strip())

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
                # pretrained_vocab, pretrained_embs = self._load_pretrained_senna()
                pretrained_vocab, pretrained_embs = self._load_pretrained_glove()

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
                    trainable=True
                )
                train_embs = tf.get_variable(
                    name='embs_only_in_train',
                    shape=[len(only_in_train), self.word_embed_size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=True
                )
                unk_embs = tf.get_variable(
                    name='embs_unk',
                    shape=[1, self.word_embed_size],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=True
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

            self.word_embedding_layer = tf.nn.embedding_lookup(word_embeddings, word_string_tensor)

            char_embeddings = tf.get_variable(
                name='embs_char',
                shape=[len(train_char_vocab) + 1, self.char_embed_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True
            )
            vocab_lookup = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(train_char_vocab),
                default_value=len(train_char_vocab)
            )
            char_string_tensor = vocab_lookup.lookup(self.chars)

            self.char_embedding_layer = tf.nn.embedding_lookup(char_embeddings, char_string_tensor)

    def _add_cnn(self):
        with tf.variable_scope('cnn'):
            # A dropout layer is applied before character embeddings are input to CNN
            cnn_inputs = tf.nn.dropout(self.char_embedding_layer, keep_prob=self.dropout)
            cnn_inputs = tf.reshape(cnn_inputs, (-1, self.max_word_length, self.char_embed_size, 1))

            # _filter = tf.get_variable(
            #     name="cnn_filter",
            #     shape=[self.max_word_length, self.char_embed_size, 1, self.filter_size],
            #     initializer=tf.contrib.layers.xavier_initializer(),
            #     trainable=True
            # )

            conv = tf.layers.conv2d(cnn_inputs,
                                    filters=self.filter_size,
                                    kernel_size=(3, self.char_embed_size),
                                    padding='VALID',
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool = tf.nn.max_pool(conv,
                                  [1, self.max_word_length - 2, 1, 1],
                                  [1, 1, 1, 1],
                                  'VALID')

            pool = tf.squeeze(pool, [1, 2])
            pool = tf.reshape(pool, (-1, self.max_seq_length, self.filter_size))

        self.embedding_layer = tf.concat([self.word_embedding_layer, pool], axis=2)

    def _add_rnn(self):
        def rnn_cell(gru=True):
            if gru:
                cell = tf.contrib.rnn.GRUCell(self.hidden_size,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
            return cell

        # dropout layers are applied on both the input and output vectors of BiLSTM
        rnn_inputs = tf.nn.dropout(self.embedding_layer, keep_prob=self.dropout)

        with tf.variable_scope('recurrent'):
            fw_cells = [rnn_cell(False) for _ in range(1)]
            bw_cells = [rnn_cell(False) for _ in range(1)]
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells,
                rnn_inputs,
                dtype=tf.float32,
                sequence_length=self.length
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
            self.unary_potentials, self.trans_params, tf.cast(self.length, tf.int32)
        )
        self.loss = tf.reduce_sum(-self.ll)

    def _add_train_op(self):
        '''
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.global_step,
                                                   878,
                                                   0.95,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        '''
        optimizer = tf.train.AdamOptimizer(0.001)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
        self._train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params),
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
            self.dropout: self._dropout
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
        """
        score: [seq_len, num_tags]
        transition_params: [num_tags, num_tags]
        """

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
