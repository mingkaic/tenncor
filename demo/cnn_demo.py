# logic source: https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html (warning: disable javascript)
# layer_defs = [];
# layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
# layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
# layer_defs.push({type:'pool', sx:2, stride:2}); // max pooling
# layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
# layer_defs.push({type:'pool', sx:2, stride:2});
# layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
# layer_defs.push({type:'pool', sx:2, stride:2});
# layer_defs.push({type:'softmax', num_classes:10});

# net = new convnetjs.Net();
# net.makeLayers(layer_defs);

# trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});

import sys
import time
import argparse

import tenncor as tc

from extenncor.tfdstrainer import TfdsEnv

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

prog_description = 'Demo cnn model'

cifar_name = 'cifar10'
assert cifar_name in tfds.list_builders()

def cross_entropy_loss(Label, Pred):
    return -tc.api.reduce_sum(Label * tc.api.log(Pred + np.finfo(float).eps), set([0]))

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
        type=str2bool, nargs='?', const=False, default=True,
        help='Whether to seed or not (default: True)')
    parser.add_argument('--seedval', dest='seedval', type=int, nargs='?', default=int(default_ts),
        help='Random seed value (default: <current time>)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='models/cnn.onnx',
        help='Filename to load pretrained model (default: models/cnn.onnx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        tc.seed(args.seedval)
        np.random.seed(args.seedval)

    names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]
    nbatch = 4
    learning_rate = 0.01
    l2_decay = 0.0001
    show_every_n = 5
    nepochs = 10

    ds = tfds.load('cifar10', split='train', batch_size=nbatch, shuffle_files=True)
    cifar = tfds.as_numpy(ds)

    # batch, height, width, in
    raw_inshape = list(ds.output_shapes['image'][1:])
    train_outshape = [nbatch, 10]

    train_in = tc.EVariable([nbatch] + raw_inshape, label="train_in")
    train_exout = tc.EVariable(train_outshape, label="train_exout")

    # construct CNN
    padding = ((2, 2), (2, 2))
    model = tc.api.layer.link([ # minimum input shape of [1, 32, 32, 3]
        tc.api.layer.bind(lambda x: x / 255. - 0.5), # normalization
        tc.api.layer.conv([5, 5], 3, 16,
            weight_init=tc.api.layer.norm_xavier_init(0.5),
            zero_padding=padding), # outputs [nbatch, 32, 32, 16]
        tc.api.layer.bind(tc.api.relu),
        tc.api.layer.bind(lambda x: tc.api.nn.max_pool2d(x, [1, 2]),
            inshape=tc.Shape([1, 32, 32, 16])), # outputs [nbatch, 16, 16, 16]
        tc.api.layer.conv([5, 5], 16, 20,
            weight_init=tc.api.layer.norm_xavier_init(0.3),
            zero_padding=padding), # outputs [nbatch, 16, 16, 20]
        tc.api.layer.bind(tc.api.relu),
        tc.api.layer.bind(lambda x: tc.api.nn.max_pool2d(x, [1, 2]),
            inshape=tc.Shape([1, 16, 16, 20])), # outputs [nbatch, 8, 8, 20]
        tc.api.layer.conv([5, 5], 20, 20,
            weight_init=tc.api.layer.norm_xavier_init(0.1),
            zero_padding=padding), # outputs [nbatch, 8, 8, 20]
        tc.api.layer.bind(tc.api.relu),
        tc.api.layer.bind(lambda x: tc.api.nn.max_pool2d(x, [1, 2]),
            inshape=tc.Shape([1, 8, 8, 20])), # outputs [nbatch, 4, 4, 20]

        tc.api.layer.dense([4, 4, 20], [10], # weight has shape [10, 4, 4, 20]
            weight_init=tc.api.layer.norm_xavier_init(0.5),
            bias_init=tc.api.layer.zero_init(),
            dims=[[0, 1], [1, 2], [2, 3]]), # outputs [nbatch, 10]
        tc.api.layer.bind(lambda x: tc.api.softmax(x, 0, 1))
    ], train_in)
    untrained = model.deep_clone()
    trained = model.deep_clone()
    try:
        print('loading ' + args.load)
        trained = tc.load_from_file(args.load)[0]
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))

    opt = lambda error, leaves: \
        tc.api.approx.adadelta(error, leaves, step_rate=learning_rate, decay=l2_decay)

    train_err = tc.apply_update([model], opt,
        lambda models: cross_entropy_loss(train_exout, models[0]))

    test_in = tc.EVariable([1] + raw_inshape, label="test_in")
    test_exout = tc.EVariable([10], label='test_exout')
    test_prob = model.connect(test_in)
    test_idx = tc.api.argmax(test_prob)
    test_err = cross_entropy_loss(test_exout, test_prob)

    test_ds = tfds.load('cifar10', split='test', batch_size=1, shuffle_files=True)

    test_gen = tfds.as_numpy(test_ds)
    def test_iteration(show=False):
        test_sample = next(test_gen)
        labels = np.zeros(10)
        labels[test_sample['label'][0]] = 1

        test_in.assign(test_sample['image'].astype(np.float))
        test_exout.assign(labels.astype(np.float))
        print('expect: {}'.format(names[test_sample['label'][0]]))
        print('got: {}'.format(names[int(test_idx.get())]))
        print('probability: {}'.format(test_prob.get()))
        print('err: {}'.format(test_err.get()))
        if show:
            plt.imshow(test_sample['image'].reshape(*raw_inshape))
            plt.show()

    backup_every = 50
    test_every = 10
    terrs = []
    def cifar_trainstep(train_idx, sess, data, trainins, trainouts):
        invar, exout = tuple(trainins)
        trainerr, trainprob = tuple(trainouts)

        labels = np.zeros((nbatch, 10))
        for j, label in enumerate(data['label']):
            labels[j][label] = 1
        print('expected label:\n{}'.format(labels))
        invar.assign(data['image'].astype(np.float))
        exout.assign(labels.astype(np.float))
        epoch_errs = []
        for j in range(nepochs):
            prob = trainprob.get()
            err = trainerr.get()
            epoch_errs.append(np.average(err))
            print('==== epoch {} ===='.format(j))
            try:
                epoch_debug()
            except:
                pass

            print('prob: {}'.format(prob))
            outof = np.sum(prob, axis=1)
            assert(np.all([s > 0.95 for s in outof]))
            print('error: {}'.format(err))

        min_err = epoch_errs[-1]
        terrs.append(min_err)
        if train_idx % show_every_n == show_every_n - 1:
            print('==== {}th image ====\nmin error:\n{}'.format(train_idx, min_err))

        if train_idx % test_every == test_every - 1:
            test_iteration()

    def error_connect(input_vars):
        opt = lambda error, leaves: tc.api.approx.adadelta(error, leaves, step_rate=learning_rate, decay=l2_decay)
        invar, exout = tuple(input_vars)

        output_prob = model.connect(invar)
        train_err = tc.apply_update([output_prob], opt,
            lambda models: cross_entropy_loss(exout, models[0]))

        return [train_err, output_prob]

    env = TfdsEnv('cifar10', [train_in, train_exout],
                error_connect, cifar_trainstep,
                split='train', batch_size=nbatch,
                display_name='demo_cifar10',
                optimize_cfg='cfg/optimizations.json')

    # train
    ntraining = 50
    for _ in range(ntraining):
        if not env.train():
            break

    plt.plot(list(range(len(terrs))), terrs)

    # test
    tds = tfds.load('cifar10', split='test', batch_size=1, shuffle_files=True)
    image = next(tfds.as_numpy(tds))
    test_in.assign(image['image'].astype(np.float))
    print('expect: {}'.format(names[image['label'][0]]))
    print('got: {}'.format(names[int(test_idx.get())]))
    print('probability: {}'.format(test_prob.get()))

    plt.imshow(image['image'].reshape(*raw_inshape))
    plt.show()

    try:
        print('saving')
        if tc.save_to_file(args.save, [model]):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if '__main__' == __name__:
    main(sys.argv[1:])
