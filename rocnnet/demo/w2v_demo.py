# source: https://towardsdatascience.com/an-implementation-guide-to-word2vec-using-numpy-and-google-sheets-13445eebd281

import numpy as np
from collections import defaultdict

import eteq.tenncor as tc
import eteq.eteq as eteq

text = "natural language processing and machine learning is fun and exciting"

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

class Model:
    def __init__(self, getW1, getW2):
        self.w1 = eteq.variable(np.array(getW1), 'w1')
        self.w2 = eteq.variable(np.array(getW2), 'w2')

    def forward(self, x):
        return tc.softmax(tc.matmul(tc.matmul(x, self.w1), self.w2))

    def backprop(self, e, h, x):
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
        # Going backwards, we need to take derivative of E with respect of w2
        # h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
        dl_dw2 = np.outer(h, e)
        # x - shape 1x8, w2 - 5x8, e.T - 8x1
        # x - 1x8, np.dot() - 5x1, dl_dw1 - 8x5
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        # Update weights
        self.w1 = self.w1 - (lr * dl_dw1)
        self.w2 = self.w2 - (lr * dl_dw2)

# training
# Initialising weight matrices
# Both s1 and s2 should be randomly initialised but for this demo, we pre-determine the arrays (getW1 and getW2)
# getW1 - shape (9x10) and getW2 - shape (10x9)
np.random.seed(0)

model = Model(
    np.random.uniform(-1, 1, (len(word_index), n)),
    np.random.uniform(-1, 1, (n, len(word_index))))

sess = eteq.Session()

winput = eteq.variable(np.random.rand(len(word_index)) * 2 - 1, 'input')
woutput = eteq.variable(np.random.rand(2 * window, len(word_index)) * 2 - 1, 'output')
y_pred = model.forward(winput)
err = tc.reduce_sum(tc.pow(tc.extend(y_pred, [1, 2 * window]) - woutput, 2.))
dw1 = eteq.derive(err, model.w1)
dw2 = eteq.derive(err, model.w2)
u1 = model.w1 - dw1 * lr
u2 = model.w2 - dw2 * lr
sess.track([u1, u2, err])

# Cycle through each epoch
for i in range(epochs):
    # Intialise loss to 0
    loss = 0

    # Cycle through each training sample
    # w_t = vector for target word, w_c = vectors for context words
    for w_t, w_c in training_data:
        wcdata = np.array(w_c)
        for j in range(2 * window - wcdata.shape[0]):
            wcdata = np.concatenate((wcdata, y_pred.get().reshape(1, len(word_index))), 0)
        winput.assign(np.array(w_t))
        woutput.assign(wcdata)
        sess.update_target([u1, u2, err])
        model.w1.assign(u1.get())
        model.w2.assign(u2.get())
        loss += err.get()
    print('Epoch:', i, "Loss:", loss)

def word_vec(word):
    w_index = word_index[word]
    v_w = model.w1.get()[w_index]
    return v_w

print(word_vec("machine"))

def vec_sim(word, top_n):
    v_w1 = word_vec(word)
    word_sim = {}

    for i in range(len(word_index)):
        # Find the similary score for each word in vocab
        v_w2 = model.w1.get()[i]
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
