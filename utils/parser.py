# conding: utf-8

import pickle
import numpy as np

from collections import Counter


def parse_conll2003():
    train_file = open('data/train.txt')
    val_file = open('data/valid.txt')
    test_file = open('data/test.txt')

    word_counter = Counter()
    char_counter = Counter()
    label_counter = Counter()

    files = [('train', train_file), ('val', val_file), ('test', test_file)]
    dump = []

    for prefix, file in files:
        x = []
        ch = []
        la = []

        sx = []  # sentence x
        sc = []  # sentence char
        sl = []  # sentence label
        for row in file:
            if row == '\n':
                if sx:
                    x.append(sx)
                    ch.append(sc)
                    la.append(sl)

                    sx = []
                    sc = []
                    sl = []
            else:
                data = row.split(' ')
                token = data[0].strip()
                label = data[-1].strip()

                sx.append(token)
                sc.append([ch for ch in token])
                sl.append(label)

                word_counter.update([token])
                char_counter.update([ch for ch in token])
                label_counter.update([label])
        dump.append([x, ch, la])

    w2idx = {key: idx for idx, key in enumerate(word_counter.keys())}
    ch2idx = {key: idx for idx, key in enumerate(char_counter.keys())}
    la2idx = {key: idx for idx, key in enumerate(label_counter.keys())}

    for i in range(3):
        dump[i][0] = [np.array([w2idx[w] for w in sw]) for sw in dump[i][0]]  # x
        dump[i][1] = [[[ch2idx[ch] for ch in ssc] for ssc in sc] for sc in dump[i][1]]  # char
        dump[i][2] = [np.array([la2idx[la] for la in sl]) for sl in dump[i][2]]  # label

    with open('data/conll.pkl', 'wb') as fp:
        pickle.dump((dump[0], dump[1], dump[2],
            {
            'words2idx': w2idx,
            'chars2idx': ch2idx,
            'labels2idx': la2idx
            }), fp)
