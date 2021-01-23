import numpy as np

import tenncor as tc

class Embedding(object):
    def __init__(self, vweight, wweight, words):
        nwords, vecsize = tuple(vweight.shape())
        self.embedding = tc.api.layer.dense([nwords], [vecsize],
            kernel_init=lambda shape, weight: vweight,
            bias_init=None)
        self.exbedding = tc.api.layer.dense([vecsize], [nwords],
            kernel_init=lambda shape, weight: wweight,
            bias_init=None)
        self.weight = vweight
        self.idx2word = words
        self.word2idx = dict([(word, i) for i, word in enumerate(words)])

    def __getitem__(self, idx):
        w = self.weight.get()
        if idx >= len(w):
            return None
        return w[idx]

    def __len__(self):
        return self.weight.shape()[0]

    def get_vec(self, word):
        if word not in self.word2idx:
            return None
        idx = self.word2idx[word]
        return self[idx]

    def onehot(self, word):
        if word not in self.word2idx:
            return None
        word_vec = [0] * len(self.idx2word)
        word_vec[self.word2idx[word]] = 1
        return word_vec

def embedding_kernel_init(shape, label):
    if isinstance(shape, tc.Shape):
        shape = shape.as_list()
    return tc.variable(np.random.uniform(
        -1, 1, tuple(shape)), label)

def make_embedding(words, vecsize):
    '''
    words      list of strings with index denoting label
    vecsize     int denoting length of mapped vector
    '''
    nwords = len(words)
    vweight = embedding_kernel_init([nwords, vecsize], 'to_vec')
    wweight = embedding_kernel_init([vecsize, nwords], 'to_word')
    return Embedding(vweight, wweight, words)

def vdistance(v1, v2):
    theta_sum = np.dot(v1, v2)
    theta_den = np.linalg.norm(v1) * np.linalg.norm(v2)
    return theta_sum / theta_den
