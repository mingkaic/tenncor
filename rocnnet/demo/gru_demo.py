# logic source: https://github.com/erikvdplas/gru-rnn
import time

import numpy as np

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

def one_encode(indices, vocab_size):
    encoding = []
    for i in indices:
        enc = np.zeros((vocab_size))
        enc[i] = 1
        encoding.append(enc)
    return np.array(encoding)

def sample(sess, inp, prob, seed_ix, n):
    # Initialize first word of sample ('seed') as one-hot encoded vector.
    x = np.zeros(inp.shape())
    x[seed_ix] = 1
    ixes = [seed_ix]

    for _ in range(n):
        inp.assign(x.T)
        sess.update_target([prob])
        p = prob.get()

        # Choose next char according to the distribution
        ix = np.random.choice(range(p.shape[0]), p=p.ravel())
        x = np.zeros(inp.shape())
        x[ix] = 1
        ixes.append(ix)

    return ixes

def old_winit(shape, label):
    return eteq.variable(np.random.uniform(-0.05, 0.05, shape.as_list()), label)

def main():
    # Seed random
    np.random.seed(0)

    # Read data and setup maps for integer encoding and decoding.
    data = open('models/data/gru_input.txt', 'r').read()
    chars = sorted(list(set(data))) # Sort makes model predictable (if seeded).
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    # Hyper parameters
    h_size, o_size, N = vocab_size, vocab_size, vocab_size # Hidden size is set to vocab_size, assuming that level of abstractness is approximately proportional to vocab_size (but can be set to any other value).
    seq_length = 25 # Longer sequence lengths allow for lengthier latent dependencies to be trained.
    learning_rate = 1e-1

    print_interval = 100

    model = rcn.SequentialModel("model")
    model.add(rcn.GRU(h_size, N,
        weight_init=old_winit,
        bias_init=rcn.zero_init(),
        label="gru"))
    model.add(rcn.Dense(o_size, eteq.Shape([h_size]),
        weight_init=old_winit,
        bias_init=rcn.zero_init(),
        label="labeller"))
    model.add(rcn.softmax(0))

    sess = eteq.Session()

    sample_inp = eteq.Variable([1, vocab_size], 0)
    sample_prob = model.connect(sample_inp)
    sess.track([sample_prob])

    inps = eteq.Variable([seq_length, vocab_size], 0)
    expected_output = eteq.Variable([seq_length, vocab_size], 0)
    trainer = rcn.sgd_train(model, sess, inps, expected_output,
        update=rcn.get_adagrad(
            learning_rate=learning_rate, epsilon=1e-8),
        errfunc=lambda encoded_expect, encoded_result:\
            tc.reduce_sum(-tc.log(tc.reduce_sum(encoded_result * encoded_expect, 0, 1))))
    eteq.optimize(sess, eteq.parse_optrules("cfg/optimizations.rules"))

    smooth_loss = -np.log(1.0/vocab_size)*seq_length
    p = 0
    start = time.time()
    for n in range(4001):
        # Reset memory if appropriate
        if p + seq_length + 1 >= len(data) or n == 0:
            p = 0

        # Get input and target sequence
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        encoded_inp = one_encode(inputs, vocab_size)
        encoded_out = one_encode(
            [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]], vocab_size)

        # Occasionally sample from oldModel and print result
        if n % print_interval == 0:
            sample_ix = sample(sess, sample_inp, sample_prob, inputs[0], 1000)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n%s\n----' % (txt, ))

        # Get gradients for current oldModel based on input and target sequences
        inps.assign(encoded_inp)
        expected_output.assign(encoded_out)
        loss = trainer().as_numpy()

        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Occasionally print loss information
        if n % print_interval == 0:
            print('iter %d, loss: %f, smooth loss: %f' % (n, loss, smooth_loss))
            print('batch training time: {} seconds'.format(time.time() - start))
            start = time.time()

        # Prepare for next iteration
        p += seq_length

if __name__ == "__main__":
    main()
