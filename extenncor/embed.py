import numpy as np

import tenncor as tc

def embedding_init(shape, label):
    return tc.variable(np.random.uniform(
        -1, 1, tuple(shape.as_list())), label)

def make_embedding(nwords, ndims):
    embedding = tc.api.layer.dense([nwords], [ndims],
        weight_init=embedding_init, bias_init=None)
    model = tc.api.layer.link([
        embedding,
        tc.api.layer.dense([ndims], [nwords],
            weight_init=embedding_init, bias_init=None),
        tc.api.layer.bind(tc.api.softmax),
    ])
    weight = embedding.get_storage()[0]
    return weight, model
