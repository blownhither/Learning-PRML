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
BATCH_SIZE = 128
EMBEDDING_SIZE = 128
SKIP_WINDOW = 1
N_SKIPS = 2

VALID_SIZE = 16
VALID_WINDOW = 100
VALID_EXAMPLES = np.random.choice(VALID_WINDOW, VALID_SIZE, replace=False)
N_SAMPLED = 64


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
                    target = np.random.randint(0, span)
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

    # define graph
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(VALID_EXAMPLES, dtype=tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1, 1))
            embedded = tf.nn.embedding_lookup(embeddings, train_inputs)
            nce_weight = tf.Variable(
                tf.truncated_normal([VOCAB_SIZE, EMBEDDING_SIZE], stddev=1.0 / np.sqrt(EMBEDDING_SIZE))
            )
            nce_biases = tf.Variable(tf.zeros([VOCAB_SIZE]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=embedded,
                                             num_sampled=N_SAMPLED,
                                             num_classes=VOCAB_SIZE))

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        init = tf.global_variables_initializer()

        # start training
        N_STEPS = 100000
        with tf.Session(graph=graph) as session:
            init.run()
            print("Initialized")

            average_loss = 0
            for step in range(N_STEPS):
                batch_inputs, batch_labels = batch_maker.get(BATCH_SIZE, N_SKIPS, SKIP_WINDOW)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    print("step %d, loss %g" % (step, average_loss / 2000))
                    average_loss = 0

                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(VALID_SIZE):
                        valid_word = reverse_dictionary[VALID_EXAMPLES[i]]
                        top_k = 2
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearset to %s:" % valid_word
                        for k in range(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str += " " + close_word
                        print(log_str)

            final_embeddings = normalized_embeddings.eval()
            print('Final embeddings %g' % final_embeddings)




if __name__ == '__main__':
    run()

