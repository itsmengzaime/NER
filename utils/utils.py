from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import bottleneck as bn

from utils.conlleval import evaluate_conll_file


def string_to_array(record):
    """Convert a record string to numpy array.
    
    record: string in the form of 
        token1  pos1  other1 ... label1
        token2  pos2  other2 ... label2
        ...
        tokenk  posk  otherk ... labelk
    """
    return np.array([x.strip().split() for x in record.strip().split('\n')])


def conll_format(token, la_true, la_pred, idx2w, idx2la, prefix):
    with open('dev/%s.predict_conll' % prefix, 'w') as fp:
        for sw, se, sl in zip(token, la_true, la_pred):
            for a, b, c in zip(sw, se, sl):
                fp.write(idx2w[a] + ' ' + idx2la[b] + ' ' + idx2la[c] + '\n')
            fp.write('\n')

    with open('dev/%s.predict_conll' % prefix, 'r') as fp:
        (prec, rec, f1) = evaluate_conll_file(fp)

    with open('eval/%s.detail' % prefix, 'w') as fp:
        fp.write('Precision: %f, Recall: %f, F1: %f\n' % (prec, rec, f1))


def viterbi_decode_topk(score, transition_params, topK = 1):
    """Decode the top K scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        k: Top K

    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """

    seq_len, num_tags = score.shape

    trellis = np.zeros((topK, seq_len, num_tags))
    backpointers = np.zeros_like(trellis, dtype=np.int32)
    trellis[0, 0] = score[0]
    trellis[1:topK, 0] = -1e16 # Mask

    # Compute score
    for t in range(1, seq_len):
        v = np.zeros((num_tags * topK, num_tags))
        for k in range(topK):
            tmp = np.expand_dims(trellis[k, t - 1], 1) + transition_params
            v[k * num_tags: (k + 1) * num_tags, :] = tmp

        args = np.argsort(-v, 0) # Desc
        for k in range(topK):
            trellis[k, t] = score[t] + v[args[k, :], np.arange(num_tags)]
            backpointers[k, t] = args[k, :]

    # Decode topK
    v = trellis[:, -1, :] # [topK, num_tags]
    v = v.flatten()

    args = np.argsort(-v)[:topK]
    scores = v[args]

    sequences = []
    for k in range(topK):
        viterbi = [args[k]]

        for t in range(seq_len - 1, 0, -1):
            last = viterbi[-1]
            id1 = last // num_tags
            id2 = last % num_tags
            viterbi.append(backpointers[id1, t, id2])

        viterbi.reverse()
        viterbi = [x % num_tags for x in viterbi]
        sequences.append(viterbi)

    return sequences, scores


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


class FeatureExtractor(object):
    def __init__(self):
        self.feat_template = {}
        self.feat_vocabs = {}
        self.feat2idx = {}

    def parse_template(self, template_filepath):
        """ Parse crf++ template file

        feat_template is in the form of 
        { 'U00': [(-1, 0)], 'U01': [(0, 0)], 'U02': [(1, 0)],
          'U03': [(-1, 0), (0, 0), (1, 0)], ... }

        """

        def _parse_line(line):
            if line.startswith('U'):
                feat_name, feat = line.strip().split(':')
                feat_comb = feat.strip().split('/')
                feat_comb = [list(map(int, x[3:-1].split(','))) for x in feat_comb]
                return feat_name, feat_comb
            else:
                return None, None

        with open(template_filepath) as f:
            for line in f:
                feat_name, feat_comb = _parse_line(line)
                if feat_name is None:
                    continue
                else:
                    self.feat_template[feat_name] = feat_comb

    def construct_vocabs_from_data(self, train_filepath, min_freq=1):
        if len(self.feat_template) == 0:
            print("No feature template parsed!")
        else:
            with open(train_filepath) as f:
                data = f.read().strip().split('\n\n')
            feat_cntrs = {}
            for feat_name, feat_comb in self.feat_template.items():
                cntr = {}
                for record in data:
                    record_arr = string_to_array(record)
                    for i in range(len(record_arr)):
                        current_feat = []
                        for x, y in feat_comb:
                            x += i  # relative position to absolute position in the array
                            if x < 0 or x >= len(record_arr):
                                current_feat.append('_PAD')
                            else:
                                current_feat.append(record_arr[x, y])
                        current_feat = ' '.join(current_feat)
                        cntr[current_feat] = cntr.get(current_feat, 0) + 1

                feat_cntrs[feat_name] = cntr

            for feat_name in self.feat_template:
                cntr = feat_cntrs[feat_name]
                self.feat_vocabs[feat_name] = sorted(
                    [x for x in cntr if cntr[x] >= min_freq],
                    key=cntr.get,
                    reverse=True
                )

            offset = 0
            for feat_name, vocab in self.feat_vocabs.items():
                self.feat2idx[feat_name] = dict(
                    (x, i + offset) for (i, x) in enumerate(vocab)
                )
                offset += len(vocab)

    def save_vocabs(self, save_dir):
        if len(self.feat_vocabs) == 0:
            print("Feature vocabs not constructed!")
        else:
            for feat_name, vocab in self.feat_vocabs.items():
                save_path = os.path.join(save_dir, feat_name + '.vocab')
                with open(save_path, 'w') as f:
                    f.write('\n'.join(vocab))

    def construct_vocabs_from_file(self, load_dir):
        if len(self.feat_template) == 0:
            print("No feature template parsed!")
        else:
            for feat_name in self.feat_template:
                load_path = os.path.join(load_dir, feat_name + '.vocab')
                if not os.path.exists(load_path):
                    print("Vocab {0} not found in {1}".format(feat_name, load_dir))
                    break
            else:
                for feat_name in self.feat_template:
                    load_path = os.path.join(load_dir, feat_name + '.vocab')
                    with open(load_path) as f:
                        self.feat_vocabs[feat_name] = [x.strip() for x in f.readlines()]

                offset = 0
                for feat_name, vocab in self.feat_vocabs.items():
                    self.feat2idx[feat_name] = dict(
                        (x, i + offset) for (i, x) in enumerate(vocab)
                    )
                    offset += len(vocab)

    @property
    def feat_size(self):
        if len(self.feat2idx) == 0:
            return 0
        else:
            return sum([len(x) for _, x in self.feat2idx.items()])

    def extract_features(self, data_filepath):
        if self.feat_size == 0:
            print("Feature vocabs not constructed!")
            return None
        else:
            with open(data_filepath) as f:
                data = f.read().strip().split('\n\n')
            results = []
            for record in data:
                record_arr = string_to_array(record)
                feat_ids = []
                for i in range(len(record_arr)):
                    step_feat_ids = []
                    for feat_name, feat_comb in self.feat_template.items():
                        current_feat = []
                        for x, y in feat_comb:
                            x += i  # relative position to absolute position in the array
                            if x < 0 or x >= len(record_arr):
                                current_feat.append('_PAD')
                            else:
                                current_feat.append(record_arr[x, y])
                        current_feat = ' '.join(current_feat)
                        feat_id = self.feat2idx[feat_name].get(current_feat, None)
                        if feat_id is not None:
                            step_feat_ids.append(feat_id)
                    feat_ids.append(step_feat_ids)
                results.append(feat_ids)
            return results


class CRFeeder(object):
    """Helper for feed train data

    Attributes:
        feats: Features [train_size, feature_size]
        labels: labels [train_size]
    """

    def __init__(self, feats, labels, max_length: int, feat_size: int, batch_size: int):
        self._feats = feats
        self._labels = labels
        self._max_length = max_length
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
        tmp = list(zip(self._feats, self._labels))
        random.shuffle(tmp)
        self._feats, self._labels = zip(*tmp)

    def feed(self):
        next_offset = min(self.size, self.offset + self._batch_size)
        batch_size = next_offset - self.offset
        feats = self._feats[self.offset: next_offset]
        labels = self._labels[self.offset: next_offset]
        self.offset = next_offset

        '''
        Should change X to (indices, values, shape)
        Should change y to (batch_size, max_length)
        '''

        shape = np.array([batch_size, self._max_length, self._feat_size])
        indices = [
            [idx1, idx2, v3]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
            for idx3, v3 in enumerate(v2)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        labels = list(map(lambda x: np.pad(x, (0, self._max_length - x.shape[0]), 'constant', constant_values=0),
                          labels))  # Pad zeors
        labels = np.array(labels, dtype=np.int32)

        return (indices, values, shape), labels

    def predict(self, feats):
        '''
        :param feats: [max_length, feat_size]
        :return: (indices, values, shape), len
        '''

        length = len(feats)

        shape = np.array([1, self._max_length, self._feat_size])
        indices = [
            [0, idx1, v2]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        return (indices, values, shape), length

    def test(self, feats):
        '''
        :param X: [batch_size, max_length, feat_size]
        :return: (indices, values, shape), len
        '''

        shape = np.array([len(feats), self._max_length, self._feat_size])
        indices = [
            [idx1, idx2, v3]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
            for idx3, v3 in enumerate(v2)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        return (indices, values, shape)


class LSTMCRFeeder(object):
    """Helper for feed train data

    Attributes:
        feats: Features [train_size, feature_size]
        labels: labels [train_size]
    """

    def __init__(self, tokens, feats, labels, max_length: int, feat_size: int, batch_size: int):
        self._tokens = tokens
        self._feats = feats
        self._labels = labels
        self._max_length = max_length
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
        feats = self._feats[self.offset: next_offset]
        labels = self._labels[self.offset: next_offset]
        self.offset = next_offset

        '''
        Change tokens to (batch_size, max_length)
        Change feats to (indices, values, shape)
        Change labels to (batch_size, max_length)
        '''

        tokens = list(map(lambda x: np.pad(x, (0, self._max_length - x.shape[0]), 'constant', constant_values=0),
                          tokens))  # Pad zeors
        tokens = np.array(tokens, dtype=np.int32)

        shape = np.array([batch_size, self._max_length, self._feat_size])
        indices = [
            [idx1, idx2, v3]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
            for idx3, v3 in enumerate(v2)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        labels = list(map(lambda x: np.pad(x, (0, self._max_length - x.shape[0]), 'constant', constant_values=0),
                          labels))  # Pad zeors
        labels = np.array(labels, dtype=np.int32)

        return tokens, (indices, values, shape), labels

    def predict(self, tokens, feats):
        '''
        :param tokens: [length]
        :param feats: [max_length, feat_size]
        :return: (indices, values, shape), len
        '''

        length = len(tokens)

        tokens = np.pad(tokens, (0, self._max_length - tokens.shape[0]), 'constant', constant_values=0)
        tokens = np.expand_dims(tokens, 0)

        shape = np.array([1, self._max_length, self._feat_size])
        indices = [
            [0, idx1, v2]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        return tokens, (indices, values, shape), length

    def test(self, tokens, feats):
        '''
        :param tokens: [batch_size, max_length]
        :param feats: [batch_size, max_length, feat_size]
        :return: (indices, values, shape), len
        '''

        tokens = list(map(lambda x: np.pad(x, (0, self._max_length - x.shape[0]), 'constant', constant_values=0),
                          tokens))  # Pad zeors
        tokens = np.array(tokens, dtype=np.int32)

        shape = np.array([len(feats), self._max_length, self._feat_size])
        indices = [
            [idx1, idx2, v3]
            for idx1, v1 in enumerate(feats)
            for idx2, v2 in enumerate(v1)
            for idx3, v3 in enumerate(v2)
        ]
        values = np.ones(len(indices))
        indices = np.array(indices)

        return tokens, (indices, values, shape)
