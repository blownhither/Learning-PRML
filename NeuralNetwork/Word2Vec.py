#!/usr/bin/env python3
# encoding: utf-8

"""
Modified from 'TensorFlow'
@license: Apache Licence
@time: 7/9/17 9:05 PM
"""
import os
from zipfile import ZipFile
from urllib import request
from collections import Counter, deque

import numpy as np
import tensorflow as tf


URL = "http://mattmahoney.net/dc/"
VOCAB_SIZE = 10000


def maybe_download(filename, n_bytes):
    if not os.path.exists(filename):
        print('Downloading...')
        filename, _ = request.urlretrieve(URL + filename, filename)
    stat = os.stat(filename)
    if stat.st_size == n_bytes:
        print(filename + "file size verified")
    else:
        print(stat.st_size)
        raise Exception("Unexpected file size")
    return filename


def read_data(filename):
    with ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    print(len(data), 'Words')
    return data


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(VOCAB_SIZE - 1))
    d = dict()
    for i, pair in enumerate(count):
        d[pair[0]] = i
    data = []
    unknown_count = 0
    for w in words:
        if w in d:
            idx = d[w]
        else:
            idx = 0
            unknown_count += 1
        data.append(idx)
    count[0][1] = unknown_count
    reverse_dict = dict(zip(d.values(), d.keys()))
    return data, count, d, reverse_dict


class W2VBatchMaker:
    def __init__(self, data):
        self.index = 0
        self.data = np.array(data)
        self.n = len(data)

    def get(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        batch = np.ndarray(shape=batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buffer = deque(maxlen=span)

        if span + self.index >= self.n:
            self._shuffle()
            self.index = 0
        buffer.extend(self.data[self.index:self.index + span])
        self.index += span

        for i in range(0, batch_size, num_skips):
            target = skip_window
            avoid = {skip_window}
            for j in range(num_skips):
                while target in avoid:
                    target = np.random.randint(0, span - 1)
                avoid.add(target)
                batch[i + j] = buffer[skip_window]
                labels[i + j] = buffer[target]
            buffer.append(self.data[self.index])
            self.index = (self.index + 1) % self.n
        return batch, labels

    def _shuffle(self):
        idx = np.arange(self.n)
        np.random.shuffle(idx)
        self.data = self.data[idx]
        self.labels = self.labels[idx]



def run():
    filename = maybe_download('text8.zip', 31344016)
    words = read_data(filename)
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    del words
    print('Most common words ', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[d] for d in data[:10]])
    batch_maker = W2VBatchMaker(data)
    ans = batch_maker.get(8, 2, 1)
    print(ans)

if __name__ == '__main__':
    run()

