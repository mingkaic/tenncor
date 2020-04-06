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

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

prog_description = 'Demo cnn model'

cifar_name = 'cifar10'
assert cifar_name in tfds.list_builders()

def cross_entropy_loss(Label, Pred):
    return -tc.reduce_sum(Label * tc.log(Pred + np.finfo(float).eps), set([0]))

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

    nbatch = 4
    learning_rate = 0.01
    l2_decay = 0.0001
    show_every_n = 5
    nepochs = 10

    ds = tfds.load('cifar10',
        split=tfds.Split.TRAIN,
        batch_size=nbatch,
        shuffle_files=True)
    cifar = tfds.as_numpy(ds)

    sess = tc.Session()

    # batch, height, width, in
    raw_inshape = list(ds.output_shapes['image'][1:])
    train_outshape = [nbatch, 10]
    print(raw_inshape)

    trainin = tc.EVariable([nbatch] + raw_inshape, label="trainin")
    trainout = tc.EVariable(train_outshape, label="trainout")

    # construct CNN
    model = tc.layer.link([ # minimum input shape of [1, 32, 32, 3]
        tc.layer.conv([5, 5], 3, 16,
            weight_init=tc.norm_xavier_init(0.5),
            zero_padding=[2, 2]), # outputs [nbatch, 32, 32, 16]
        tc.layer.bind(tc.relu),
        tc.layer.bind(lambda x: tc.nn.max_pool2d(x, [1, 2]),
            inshape=tc.Shape([1, 32, 32, 16])), # outputs [nbatch, 16, 16, 16]
        tc.layer.conv([5, 5], 16, 20,
            weight_init=tc.norm_xavier_init(0.3),
            zero_padding=[2, 2]), # outputs [nbatch, 16, 16, 20]
        tc.layer.bind(tc.relu),
        tc.layer.bind(lambda x: tc.nn.max_pool2d(x, [1, 2]),
            inshape=tc.Shape([1, 16, 16, 20])), # outputs [nbatch, 8, 8, 20]
        tc.layer.conv([5, 5], 20, 20,
            weight_init=tc.norm_xavier_init(0.1),
            zero_padding=[2, 2]), # outputs [nbatch, 8, 8, 20]
        tc.layer.bind(tc.relu),
        tc.layer.bind(lambda x: tc.nn.max_pool2d(x, [1, 2]),
            inshape=tc.Shape([1, 8, 8, 20])), # outputs [nbatch, 4, 4, 20]

        tc.layer.dense([4, 4, 20], [10], # weight has shape [10, 4, 4, 20]
            weight_init=tc.norm_xavier_init(0.5),
            bias_init=tc.zero_init(),
            dims=[[0, 1], [1, 2], [2, 3]]), # outputs [nbatch, 10]
        tc.layer.bind(lambda x: tc.softmax(x, 0, 1))
    ], trainin)

    untrained = model.deep_clone()
    trained = model.deep_clone()
    try:
        print('loading ' + args.load)
        trained = tc.load_from_file(args.load)[0]
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))

    opt = lambda error, leaves: tc.approx.adadelta(error, leaves,
        step_rate=learning_rate, decay=l2_decay)

    normalized = trainin / 255. - 0.5
    output_prob = model.connect(normalized)
    train_err = tc.sgd_train(model, normalized, trainout,
        opt, err_func=cross_entropy_loss)

    testin = tc.EVariable([1] + raw_inshape, label="testin")
    testout = model.connect(testin)
    testidx = tc.argmax(testout)

    sess.track([train_err, output_prob, testidx])
    tc.optimize(sess, "cfg/optimizations.json")

    # train
    terrs = []
    for i, data in enumerate(cifar):
        labels = np.zeros((nbatch, 10))
        for j, label in enumerate(data['label']):
            labels[j][label] = 1
        print('expected label:\n{}'.format(labels))
        trainin.assign(data['image'].astype(np.float))
        trainout.assign(labels.astype(np.float))
        epoch_errs = []
        for j in range(nepochs):
            sess.update_target([train_err, output_prob])
            trained_err = train_err.get()
            prob = output_prob.get()
            epoch_errs.append(np.average(trained_err))
            print('==== epoch {} ===='.format(j))
            print('prob: {}'.format(prob))
            outof = np.sum(prob, axis=1)
            assert(np.all([s > 0.95 for s in outof]))
            print('training error: {}'.format(trained_err))
        avg_err = epoch_errs[-1]
        if i % show_every_n == show_every_n - 1:
            print('==== {}th image ====\naverage error:\n{}'.format(i, avg_err))
        terrs.append(avg_err)

        if i > 500:
            break

    plt.plot(list(range(len(terrs))), terrs)

    # test
    tds = tfds.load('cifar10',
        split=tfds.Split.TEST,
        batch_size=1,
        shuffle_files=True)
    names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]
    image = next(tfds.as_numpy(tds))
    testin.assign(image['image'].astype(np.float))
    sess.update_target([testidx])
    print('expect: {}'.format(names[image['label'][0]]))
    print('got: {}'.format(names[int(testidx.get())]))
    print('probability: {}'.format(testout.get()))

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
