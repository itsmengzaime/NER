# coding: utf-8

import os
from collections import OrderedDict
import numpy as np


def string_to_array(record):
    """Convert a record string to numpy array.

    record: string in the form of
        token1  pos1  other1 ... label1
        token2  pos2  other2 ... label2
        ...
        tokenk  posk  otherk ... labelk
    """
    return np.array([x.strip().split() for x in record.strip().split('\n')])


class FeatureExtractor(object):
    def __init__(self):
        self.feat_template = OrderedDict()
        self.feat_vocabs = OrderedDict()
        self.feat2idx = OrderedDict()

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

