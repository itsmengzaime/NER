# coding: utf-8

import random
import numpy as np


class LSTMCNNCRFeeder(object):
    """Helper for feed train data

    Attributes:
        feats: Features [train_size, feature_size]
        labels: labels [train_size]
    """

    def __init__(self, tokens,
                 chars,
                 feats,
                 labels,
                 max_seq_length: int,
                 max_char_length: int,
                 feat_size: int,
                 batch_size: int):

        self._tokens = tokens
        self._chars = chars
        self._feats = feats
        self._labels = labels
        self._max_seq_length = max_seq_length
        self._max_char_length = max_char_length
        self._feat_size = feat_size
        self._batch_size = batch_size

        self.size = len(feats)
        self.offset = 0
        self.epoch = 1

    @property
    def step_per_epoch(self):
        return (self.size - 1) // self._batch_size + 1

    def next_epoch(self):
        self.offset = 0
        self.epoch += 1

        # Shuffle
        tmp = list(zip(self._tokens, self._feats, self._labels))
        random.shuffle(tmp)
        self._tokens, self._feats, self._labels = zip(*tmp)

    def feed(self):
        next_offset = min(self.size, self.offset + self._batch_size)
        batch_size = next_offset - self.offset  # May not be the setting batch size at the end
        tokens = self._tokens[self.offset: next_offset]
        chars = self._chars[self.offset: next_offset]
        feats = self._feats[self.offset: next_offset]
        labels = self._labels[self.offset: next_offset]
        self.offset = next_offset

        '''
        Change tokens to (batch_size, max_seq_length)
        Change chars to (batch_size, max_seq_length, max_char_length)
        Change feats to (indices, values, shape)
        Change labels to (batch_size, max_length)
        '''

        tokens = list(map(lambda x: np.pad(x, (0, self._max_seq_length - x.shape[0]), 'constant', constant_values=0),
                          tokens))  # Pad zeors
        tokens = np.array(tokens, dtype=np.int32)

        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = np.pad(chars[i][j], (0, self._max_char_length - len(chars[i][j])), 'constant', constant_values=0)
            for j in range(self._max_seq_length - len(chars[i])):
                chars[i].append(np.zeros((self._max_char_length,), dtype=np.int32))
        chars = np.array(chars, dtype=np.int32)

        shape = np.array([batch_size, self._max_seq_length, self._feat_size])
        indices = [
            [idx1, idx2, v3]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
            for idx3, v3 in enumerate(v2)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        labels = list(map(lambda x: np.pad(x, (0, self._max_seq_length - x.shape[0]), 'constant', constant_values=0),
                          labels))  # Pad zeors
        labels = np.array(labels, dtype=np.int32)

        return tokens, chars, (indices, values, shape), labels

    def predict(self, tokens, chars, feats):
        '''
        :param tokens: [length]
        :param feats: [max_length, feat_size]
        :return: (indices, values, shape), len
        '''

        length = len(tokens)

        tokens = np.pad(tokens, (0, self._max_seq_length - tokens.shape[0]), 'constant', constant_values=0)
        tokens = np.expand_dims(tokens, 0)

        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = np.pad(chars[i][j], (0, self._max_char_length - len(chars[i][j])), 'constant', constant_values=0)
            for j in range(self._max_seq_length - len(chars[i])):
                chars[i].append([0] * self._max_char_length)
        chars = np.array(chars, dtype=np.int32)

        shape = np.array([1, self._max_seq_length, self._feat_size])
        indices = [
            [0, idx1, v2]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        return tokens, chars, (indices, values, shape), length

    def test(self, tokens, chars, feats):
        '''
        :param tokens: [batch_size, max_length]
        :param feats: [batch_size, max_length, feat_size]
        :return: (indices, values, shape), len
        '''

        tokens = list(map(lambda x: np.pad(x, (0, self._max_seq_length - x.shape[0]), 'constant', constant_values=0),
                          tokens))  # Pad zeors
        tokens = np.array(tokens, dtype=np.int32)


        for i in range(len(chars)):
            for j in range(len(chars[i])):
                chars[i][j] = np.pad(chars[i][j], (0, self._max_char_length - len(chars[i][j])), 'constant', constant_values=0)
            for j in range(self._max_seq_length - len(chars[i])):
                chars[i].append([0] * self._max_char_length)
        chars = np.array(chars, dtype=np.int32)


        shape = np.array([len(feats), self._max_seq_length, self._feat_size])

        indices = [
            [idx1, idx2, v3]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
            for idx3, v3 in enumerate(v2)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        return tokens, chars, (indices, values, shape)