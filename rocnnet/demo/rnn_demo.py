# source: https://peterroelants.github.io/posts/rnn-implementation-part02/
import sys
import functools
import time
import argparse

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

prog_description = 'Demo rnn_trainer'

def cross_entropy_loss(Y, T):
    epsilon = 1e-5 # todo: make epsilon padding configurable for certain operators in eteq
    leftY = Y + epsilon
    rightT = 1 - Y + epsilon
    return -(T * tc.log(leftY) + (1-T) * tc.log(rightT))

def loss(T, Y):
    return tc.reduce_mean(cross_entropy_loss(Y, T))

def create_dataset(nb_samples, sequence_len):
    """Create a dataset for binary addition and
    return as input, targets."""
    max_int = 2**(sequence_len-1) # Maximum integer that can be added
     # Transform integer in binary format
    format_str = '{:0' + str(sequence_len) + 'b}'
    nb_inputs = 2  # Add 2 binary numbers
    nb_outputs = 1  # Result is 1 binary number
    # Input samples
    X = np.zeros((nb_samples, sequence_len, nb_inputs))
    # Target samples
    T = np.zeros((nb_samples, sequence_len, nb_outputs))
    # Fill up the input and target matrix
    for i in range(nb_samples):
        # Generate random numbers to add
        nb1 = np.random.randint(0, max_int)
        nb2 = np.random.randint(0, max_int)
        # Fill current input and target row.
        # Note that binary numbers are added from right to left,
        #  but our RNN reads from left to right, so reverse the sequence.
        X[i,:,0] = list(
            reversed([int(b) for b in format_str.format(nb1)]))
        X[i,:,1] = list(
            reversed([int(b) for b in format_str.format(nb2)]))
        T[i,:,0] = list(
            reversed([int(b) for b in format_str.format(nb1+nb2)]))
    return X, T

def weight_init(shape, label):
    slist = shape.as_list()
    a = np.sqrt(6.0 / np.sum(slist))
    return eteq.variable(np.array(np.random.uniform(-a, a, slist)), label)

def make_rms_prop(learning_rate, momentum_term, lmbd, eps):
    def rms_prop(grads):
        targets = [target for target, _ in grads]
        gs = [grad for _, grad in grads]
        momentum = [eteq.scalar_variable(0, target.shape()) for target in targets]
        mvavg_sqr = [eteq.scalar_variable(0, target.shape()) for target in targets]

        # group 1
        momentum_tmp = [mom * momentum_term for mom in momentum]
        group1 = [rcn.VarAssign(target, target + source) for target, source in zip(targets, momentum_tmp)]

        # group 2
        group2 = [rcn.VarAssign(target, lmbd * target + (1-lmbd) * tc.pow(grad, 2)) for target, grad in zip(mvavg_sqr, gs)]

        # group 3
        pgrad_norm_nodes = [(learning_rate * grad) / tc.sqrt(mvsqr) + eps
            for grad, mvsqr in zip(gs, mvavg_sqr)]

        group3 = [rcn.VarAssign(target, momentum_tmp_node - pgrad_norm_node)
            for target, momentum_tmp_node, pgrad_norm_node in zip(momentum, momentum_tmp, pgrad_norm_nodes)]

        group3 += [rcn.VarAssign(target, target - pgrad_norm) for target, pgrad_norm in zip(targets, pgrad_norm_nodes)]

        return [group1, group2, group3]

    return rms_prop

def rnn_train(rnn, sess, train_input, train_exout,
    learning_rate, momentum_term, lmbd, eps):

    winp, binp, _, w, b, _, _, state0, wout, bout, _, _ = tuple(rnn.get_contents())

    targets = [winp, binp, w, b, state0, wout, bout]
    momentums = [np.zeros(v.shape()) for v in targets]
    moving_avg_sqr = [np.zeros(v.shape()) for v in targets]

    train_out = rnn.connect(train_input)
    error = loss(train_exout, train_out)
    grads =  [eteq.derive(error, var) for var in targets]

    sess.track(grads)

    # apply rms prop optimization
    def train():
        # group 1
        momentum_tmp = [mom * momentum_term for mom in momentums]
        for var, mom_tmp in zip(targets, momentum_tmp):
            var.assign(np.array(var.get() + mom_tmp))

        sess.update_target(grads)
        res_grads = [grad.get() for grad in grads]

        # group 2
        for i in range(len(targets)):
            moving_avg_sqr[i] = lmbd * moving_avg_sqr[i] + (1-lmbd) * res_grads[i]**2

        # group 3
        pgrad_norms = [(learning_rate * grad) / np.sqrt(mvsqr) + eps
            for grad, mvsqr in zip(res_grads, moving_avg_sqr)]

        for i in range(len(targets)):
            momentums[i] = momentum_tmp[i] - pgrad_norms[i]

        for var, pgrad_norm in zip(targets, pgrad_norms):
            var.assign(np.array(var.get() - pgrad_norm))

    return train

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
    parser.add_argument('--n_batch', dest='n_batch', type=int, nargs='?', default=100,
        help='Batch size when training (default: 100)')
    parser.add_argument('--n_train', dest='n_train', type=int, nargs='?', default=2000,
        help='Number of times to train (default: 2000)')
    parser.add_argument('--n_test', dest='n_test', type=int, nargs='?', default=5,
        help='Number of times to test (default: 5)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='models/rnnmodel.pbx',
        help='Filename to load pretrained model (default: models/rnnmodel.pbx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        eteq.seed(args.seedval)
        np.random.seed(args.seedval)
    else:
        np.random.seed(seed=1)

    # dataset parameters
    n_train = args.n_train
    n_test = args.n_test
    sequence_len = 7

    # training parameters
    n_batch = args.n_batch
    lmbd = 0.5
    learning_rate = 0.05
    momentum_term = 0.80
    eps = 1e-6

    # create training samples
    xtrain, expect_train = create_dataset(n_train, sequence_len)
    print(f'xtrain tensor shape: {xtrain.shape}')
    print(f'expect_train tensor shape: {expect_train.shape}')

    # keep this here to get nice weights
    rcn.Dense(2, eteq.Shape([3]), weight_init)
    rcn.Dense(3, eteq.Shape([3]), weight_init)
    rcn.Dense(3, eteq.Shape([1]), weight_init)

    # model parameters
    nunits = 3  # Number of states in the recurrent layer
    ninput = 2
    noutput = 1

    model = rcn.SequentialModel("demo")
    model.add(
        rcn.Dense(nunits, eteq.Shape([ninput]),
        weight_init, label="input"))
    model.add(rcn.Recur(nunits, rcn.tanh(),
        weight_init=weight_init,
        bias_init=rcn.zero_init(), label="unfold"))
    model.add(rcn.Dense(noutput, eteq.Shape([nunits]),
        weight_init, label="output"))
    model.add(rcn.sigmoid(label="classifier"))

    untrained = model.clone()
    try:
        print('loading ' + args.load)
        trained = rcn.load_file_seqmodel(args.load, "demo")
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))
        trained = model.clone()

    sess = eteq.Session()

    train_input = eteq.Variable([n_batch, sequence_len, ninput])
    train_exout = eteq.Variable([n_batch, sequence_len, noutput])

    error = loss(train_exout, model.connect(train_input))
    sess.track([error])

    # train = rcn.sgd_train(model, sess, train_input, train_exout,
    #     make_rms_prop(learning_rate, momentum_term, lmbd, eps),
    #     errfunc=loss)
    train = rnn_train(model, sess, train_input, train_exout,
        learning_rate, momentum_term, lmbd, eps)

    test_input = eteq.Variable([n_test, sequence_len, ninput])
    untrained_out = tc.round(untrained.connect(test_input))
    trained_out = tc.round(model.connect(test_input))
    pretrained_out = tc.round(trained.connect(test_input))
    sess.track([
        untrained_out,
        trained_out,
        pretrained_out,
    ])
    eteq.optimize(sess, eteq.parse_optrules("cfg/optimizations.rules"))

    train_input.assign(xtrain[0:n_batch,:,:])
    train_exout.assign(expect_train[0:n_batch,:,:])
    sess.update_target([error])
    ls_of_loss = [error.get()]
    start = time.time()
    for i in range(5):
        for j in range(n_train // n_batch):
            xbatch = xtrain[j:j+n_batch,:,:]
            tbatch = expect_train[j:j+n_batch,:,:]

            train_input.assign(xbatch)
            train_exout.assign(tbatch)

            train()

            # Add loss to list to plot
            sess.update_target([error])
            ls_of_loss.append(error.get())
    print('training time: {} seconds'.format(time.time() - start))

    # Plot the loss over the iterations
    fig = plt.figure(figsize=(5, 3))
    plt.plot(ls_of_loss, 'b-')
    plt.xlabel('minibatch iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Loss over backprop iteration')
    plt.xlim(0, 100)
    fig.subplots_adjust(bottom=0.2)
    plt.show()

    xtest, expect_test = create_dataset(n_test, sequence_len)

    test_input.assign(xtest)
    sess.update_target([
        untrained_out,
        trained_out,
        pretrained_out,
    ])
    got_untrained = untrained_out.get()
    got_trained = trained_out.get()
    got_pretrained = pretrained_out.get()

    for i in range(xtest.shape[0]):
        left = xtest[i,:,0]
        right = xtest[i,:,1]
        expected = expect_test[i,:,:]
        yuntrained = got_untrained[i,:,:]
        ytrained = got_trained[i,:,:]
        ypretrained = got_pretrained[i,:,:]

        left = ''.join([str(int(d)) for d in left])
        left_num = int(''.join(reversed(left)), 2)
        right = ''.join([str(int(d)) for d in right])
        right_num = int(''.join(reversed(right)), 2)
        expected = ''.join([str(int(d[0])) for d in expected])
        expected_num = int(''.join(reversed(expected)), 2)
        yuntrained = ''.join([str(int(d[0])) for d in yuntrained])
        yuntrained_num = int(''.join(reversed(yuntrained)), 2)
        ytrained = ''.join([str(int(d[0])) for d in ytrained])
        ytrained_num = int(''.join(reversed(ytrained)), 2)
        ypretrained = ''.join([str(int(d[0])) for d in ypretrained])
        ypretrained_num = int(''.join(reversed(ypretrained)), 2)
        print(f'left:         {left:s}   {left_num:2d}')
        print(f'right:      + {right:s}   {right_num:2d}')
        print(f'              -------   --')
        print(f'expected:   = {expected:s}   {expected_num:2d}')
        print(f'untrained:  = {yuntrained:s}   {yuntrained_num:2d}')
        print(f'trained:    = {ytrained:s}   {ytrained_num:2d}')
        print(f'pretrained: = {ypretrained:s}   {ypretrained_num:2d}')

        print('')

    try:
        print('saving')
        if model.save_file(args.save):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if '__main__' == __name__:
    main(sys.argv[1:])
