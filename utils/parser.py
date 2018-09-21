# conding: utf-8

import pickle
import numpy as np

max_word_len = 20

def parse_conll2003():
    train_file = open('data/train.txt')
    val_file = open('data/valid.txt')
    test_file = open('data/test.txt')

    word_set = set()
    char_set = set()
    label_set = set()

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
                token = data[0].strip().lower()
                chars = ['0' if ch.isdigit() else ch for ch in token]
                label = data[-1].strip()

                if len(chars) > max_word_len:
                    half = max_word_len / 2
                    chars = chars[:half] + chars[-(max_word_len - half):]

                sx.append(token)
                sc.append(chars)
                sl.append(label)

                if prefix == 'train':
                    # Should only update word in train set
                    word_set.add(token)
                    char_set.update(*chars)
                    label_set.add(label)
        dump.append([x, ch, la])

    w2idx = {}
    ch2idx = {}
    la2idx = {}

    with open('dev/train.word.vocab', 'w') as fp:
        for idx, word in enumerate(sorted(word_set)):
            w2idx[word] = idx
            fp.write(word + '\n')

    with open('dev/train.char.vocab', 'w') as fp:
        for idx, char in enumerate(sorted(char_set)):
            ch2idx[char] = idx
            fp.write(char + '\n')

    for idx, label in enumerate(sorted(label_set)):
        la2idx[label] = idx

    for i in range(3):
        # dump[i][1] = [[[ch2idx.get(ch, len(ch2idx)) for ch in ssc] for ssc in sc] for sc in dump[i][1]]  # char
        dump[i][2] = [np.array([la2idx[la] for la in sl]) for sl in dump[i][2]]  # label

    with open('dev/conll.pkl', 'wb') as fp:
        pickle.dump((dump[0], dump[1], dump[2],
            {
            'words2idx': w2idx,
            'chars2idx': ch2idx,
            'labels2idx': la2idx
            }), fp)
