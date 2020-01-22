import numpy as np

import eteq.tenncor as tc
import eteq.eteq as eteq
import layr.layr as layr

def embedding_init(shape, label):
    return eteq.variable(np.random.uniform(
        -1, 1, tuple(shape.as_list())), label)

def make_embedding(nwords, ndims):
    embedding = layr.dense([nwords], [ndims],
        weight_init=embedding_init, bias_init=None)
    model = layr.link([
        embedding,
        layr.dense([ndims], [nwords],
            weight_init=embedding_init, bias_init=None),
        layr.bind(tc.softmax),
    ])
    weight = embedding.get_storage()[0]
    return weight, model
