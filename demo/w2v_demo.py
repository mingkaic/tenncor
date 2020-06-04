# source: https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281

import numpy as np
from collections import defaultdict

import tenncor as tc

from extenncor.embed import make_embedding

text = "natural language processing and machine learning is fun and exciting"
np.random.seed(0)

# Note the .lower() as upper and lowercase does not matter in our implementation
# [['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'and', 'exciting']]
corpus = [[word.lower() for word in text.split()]]

n = 10 # dimensions of word embeddings, also refer to size of hidden layer
lr = 0.01 # learning rate
epochs = 50 # number of training epochs
window = 2 # context window +- center word

def word2onehot(word_index, word):
    # word_vec - initialise a blank vector
    word_vec = [0 for i in range(len(word_index))] # Alternative - np.zeros(v_count)
    # Get ID of word from word_index
    word_index = word_index[word]
    # Change value from 0 to 1 according to ID of the word
    word_vec[word_index] = 1
    return word_vec

def generate_training_data(corpus):
    # Find unique word counts using dictonary
    word_counts = defaultdict(int)
    for row in corpus:
        for word in row:
            word_counts[word] += 1
    # Generate Lookup Dictionaries (vocab)
    words_list = list(word_counts.keys())
    # Generate word:index
    word_index = dict((word, i) for i, word in enumerate(words_list))
    index_word = dict((i, word) for i, word in enumerate(words_list))

    training_data = []
    # Cycle through each sentence in corpus
    for sentence in corpus:
        sent_len = len(sentence)
        # Cycle through each word in sentence
        for i, word in enumerate(sentence):
            # Convert target word to one-hot
            w_target = word2onehot(word_index, sentence[i])
            # Cycle through context window
            w_context = []
            # Note: window_size 2 will have range of 5 values
            for j in range(i - window, i + window+1):
                # Criteria for context word
                # 1. Target word cannot be context word (j != i)
                # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range
                if j != i and j <= sent_len-1 and j >= 0:
                    # Append the one-hot representation of word to w_context
                    w_context.append(word2onehot(word_index, sentence[j]))
                    # print(sentence[i], sentence[j])
                    # training_data contains a one-hot representation of the target word and context words
            training_data.append([w_target, w_context])
    return np.array(training_data), word_index, index_word

training_data, word_index, index_word = generate_training_data(corpus)

# training
# Initialising weight matrices
# Both s1 and s2 should be randomly initialised but for this demo, we pre-determine the arrays (getW1 and getW2)
# getW1 - shape (9x10) and getW2 - shape (10x9)

w1, model = make_embedding(len(word_index), n)

winput = tc.variable(np.random.rand(len(word_index)) * 2 - 1, 'input')
woutput = tc.variable(np.random.rand(2 * window, len(word_index)) * 2 - 1, 'output')

y_pred = model.connect(winput)

train_err = tc.apply_update([model],
    lambda error, leaves: tc.api.approx.sgd(error, leaves, lr),
    lambda models: tc.api.reduce_sum(tc.api.pow(tc.api.extend(models[0].connect(winput), [1, 2 * window]) - woutput, 2.)))

tc.optimize("cfg/optimizations.json")

# Cycle through each epoch
for i in range(epochs):
    # Intialise loss to 0
    loss = 0

    # Cycle through each training sample
    # w_t = vector for target word, w_c = vectors for context words
    for w_t, w_c in training_data:
        wcdata = np.array(w_c)
        ydata = y_pred.get().reshape(1, len(word_index))
        for j in range(2 * window - wcdata.shape[0]):
            wcdata = np.concatenate((wcdata, ydata), 0)
        winput.assign(np.array(w_t))
        woutput.assign(wcdata)
        loss += train_err.get()
    print('Epoch:', i, "Loss:", loss)

def word_vec(word):
    w_index = word_index[word]
    v_w = w1.get()[w_index]
    return v_w

print(word_vec("machine"))

def vec_sim(word, top_n):
    v_w1 = word_vec(word)
    word_sim = {}

    for i in range(len(word_index)):
        # Find the similary score for each word in vocab
        v_w2 = w1.get()[i]
        theta_sum = np.dot(v_w1, v_w2)
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_sum / theta_den

        word = index_word[i]
        word_sim[word] = theta

    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

    for word, sim in words_sorted[:top_n]:
        print(word, sim)

# Find similar words
vec_sim("machine", 3)
