# logic source: https://github.com/erikvdplas/gru-rnn
import sys
import time
import argparse

import numpy as np

import tenncor as tc

prog_description = 'Demo lstm model against a latin corpus'

def one_encode(indices, vocab_size):
    encoding = []
    for i in indices:
        enc = np.zeros((vocab_size))
        enc[i] = 1
        encoding.append(enc)
    return np.array(encoding)

def sample(inp, prob, seed_ix, n):
    # Initialize first word of sample ('seed') as one-hot encoded vector.
    x = np.zeros(inp.shape())
    x[seed_ix] = 1
    ixes = [seed_ix]

    for _ in range(n):
        inp.assign(x.T)
        p = prob.get()
        p /= p.sum() # normalize

        # Choose next char according to the distribution
        ix = np.random.choice(range(p.shape[0]), p=p.ravel())
        x = np.zeros(inp.shape())
        x[ix] = 1
        ixes.append(ix)

    return ixes

def encoded_loss(encoded_expect, encoded_result):
    return tc.api.reduce_sum(-tc.api.log(tc.api.reduce_sum(encoded_result * encoded_expect, 0, 1)))

def str2bool(opt):
    optstr = opt.lower()
    if optstr in ('yes', 'true', 't', 'y', '1'):
        return True
    elif optstr in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    default_ts = time.time()

    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('--seed', dest='seed',
        type=str2bool, nargs='?', const=False, default=False,
        help='Whether to seed or not (default: True)')
    parser.add_argument('--seedval', dest='seedval', type=int, nargs='?', default=int(default_ts),
        help='Random seed value (default: <current time>)')
    parser.add_argument('--n_train', dest='n_train', type=int, nargs='?', default=10001,
        help='Number of times of train (default: 10001)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='models/latin_lstm.onnx',
        help='Filename to load pretrained model (default: models/latin_lstm.onnx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        tc.seed(args.seedval)
        np.random.seed(args.seedval)
    else:
        np.random.seed(seed=0)

    # Read data and setup maps for integer encoding and decoding.
    data = open('models/data/latin_input.txt', 'r').read()
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

    def old_winit(shape, label):
        return tc.variable(np.random.uniform(-0.05, 0.05, shape.as_list()), label)

    model = tc.api.layer.link([
        tc.api.layer.lstm(tc.Shape([N]), h_size, seq_length,
            kernel_init=tc.api.init.random_uniform(-0.05, 0.05)),
        tc.api.layer.dense([h_size], [o_size],
            kernel_init=tc.api.init.random_uniform(-0.05, 0.05)),
        tc.api.layer.bind(lambda x: tc.api.softmax(x, 0, 1)),
    ])
    untrained_model = model.deep_clone()
    pretrained_model = model.deep_clone()
    try:
        print('loading ' + args.load)
        pretrained_model = tc.load_from_file(args.load)[0]
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))

    sample_inp = tc.Variable([1, vocab_size], 0)

    trained_prob = tc.api.slice(model.connect(sample_inp), 0, 1, 1)
    untrained_prob = tc.api.slice(untrained_model.connect(sample_inp), 0, 1, 1)
    pretrained_prob = tc.api.slice(pretrained_model.connect(sample_inp), 0, 1, 1)

    train_inps = tc.Variable([seq_length, vocab_size], 0)
    train_exout = tc.Variable([seq_length, vocab_size], 0)

    train_err = tc.apply_update([model],
        lambda error, leaves: tc.api.approx.adagrad(error, leaves, learning_rate=learning_rate, epsilon=1e-8),
        lambda models: encoded_loss(train_exout, models[0].connect(train_inps)))

    tc.optimize("cfg/optimizations.json")

    smooth_loss = -np.log(1.0/vocab_size)*seq_length
    p = 0
    start = time.time()
    for i in range(args.n_train):
        # Reset memory if appropriate
        if p + seq_length + 1 >= len(data) or i == 0:
            p = 0

        # Get input and target sequence
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        encoded_inp = one_encode(inputs, vocab_size)
        encoded_out = one_encode(
            [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]], vocab_size)

        # Occasionally sample from oldModel and print result
        if i % print_interval == 0:
            sample_ix = sample(sample_inp, trained_prob, inputs[0], 1000)
            print('----\n%s\n----' % (''.join(ix_to_char[ix] for ix in sample_ix)))

        # Get gradients for current oldModel based on input and target sequences
        train_inps.assign(encoded_inp)
        train_exout.assign(encoded_out)
        loss = train_err.get()

        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Occasionally print loss information
        if i % print_interval == 0:
            print('iter %d, loss: %f, smooth loss: %f' % (i, loss, smooth_loss))
            print('batch training time: {} seconds'.format(time.time() - start))
            start = time.time()

        # Prepare for next iteration
        p += seq_length

    untrained_sample = sample(sample_inp, untrained_prob, char_to_ix[data[0]], 1000)
    trained_sample = sample(sample_inp, trained_prob, char_to_ix[data[0]], 1000)
    pretrained_sample = sample(sample_inp, pretrained_prob, char_to_ix[data[0]], 1000)
    print('--untrained--\n%s\n----' % (''.join(ix_to_char[ix] for ix in untrained_sample)))
    print('--trained--\n%s\n----' % (''.join(ix_to_char[ix] for ix in trained_sample)))
    print('--pretrained--\n%s\n----' % (''.join(ix_to_char[ix] for ix in pretrained_sample)))

    try:
        print('saving')
        if tc.save_to_file(args.save, [model]):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if __name__ == "__main__":
    main(sys.argv[1:])
