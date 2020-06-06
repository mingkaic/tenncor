import numpy as np

import tenncor as tc

class Embedding(object):
    def __init__(self, model, weight, words):
        self.model = model
        self.weight = weight
        self.idx2word = words
        self.word2idx = dict([(word, i) for i, word in enumerate(words)])

    def __getitem__(self, idx):
        return self.weight.get()[idx]

    def get_vec(self, word):
        idx = self.word2idx[word]
        return self.weight.get()[idx]

    def onehot(self, word):
        word_vec = [0] * len(self.idx2word)
        word_vec[self.word2idx[word]] = 1
        return word_vec

def embedding_weight_init(shape, label):
    return tc.variable(np.random.uniform(
        -1, 1, tuple(shape.as_list())), label)

def make_embedding(words, vecsize):
    '''
    words      list of strings with index denoting label
    vecsize     int denoting length of mapped vector
    '''
    nwords = len(words)
    embedding = tc.api.layer.dense([nwords], [vecsize],
        weight_init=embedding_weight_init, bias_init=None)
    model = tc.api.layer.link([
        embedding,
        tc.api.layer.dense([vecsize], [nwords],
            weight_init=embedding_weight_init, bias_init=None),
        tc.api.layer.bind(tc.api.softmax),
    ])
    weight = embedding.get_storage()[0]
    return Embedding(model, weight, words)

def vdistance(v1, v2):
    theta_sum = np.dot(v1, v2)
    theta_den = np.linalg.norm(v1) * np.linalg.norm(v2)
    return theta_sum / theta_den
