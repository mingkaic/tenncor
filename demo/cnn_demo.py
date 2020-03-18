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
import dbg.psess as ps
import query.query as q

import tensorflow_datasets as tfds
import numpy as np

import matplotlib.pyplot as plt

prog_description = 'Demo cnn model'

cifar_name = 'cifar10'
assert cifar_name in tfds.list_builders()

def cross_entropy_loss(T, Y):
    # epsilon = 1e-5 # todo: make epsilon padding configurable for certain operators in tc
    # leftY = Y + epsilon
    # rightT = 1 - Y + epsilon
    # return -(T * tc.log(leftY) + (1-T) * tc.log(rightT))
    return tc.reduce_mean(tc.pow(T - Y, 2.))

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
    momentum = 0.9
    weight_decay = 0.0001
    show_every_n = 5
    nepochs = 10

    ds = tfds.load('cifar10',
        split=tfds.Split.TRAIN,
        batch_size=nbatch,
        shuffle_files=True)
    cifar = tfds.as_numpy(ds)

    # batch, height, width, in
    raw_inshape = [dim.value for dim in ds.output_shapes['image']]

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

        tc.layer.dense([20, 4, 4, 1], [10], # weight has shape [10, 4, 4, 20]
            weight_init=tc.norm_xavier_init(0.5),
            bias_init=tc.zero_init(),
            dims=[[0, 1], [1, 2], [2, 3]]), # outputs [nbatch, 10]
        tc.layer.bind(lambda x: tc.softmax(x, 1, 1))
    ], tc.EVariable([1, 32, 32, 3], label='input'))

    untrained = model.deep_clone()
    trained = model.deep_clone()
    try:
        print('loading ' + args.load)
        trained = tc.load_from_file(args.load)[0]
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))

    sess = ps.PluginSess()
    inspector = ps.Inspector()
    sess.add_plugin(inspector)

    raw_inshape[0] = 1
    test_inshape = raw_inshape
    testin = tc.EVariable(test_inshape, label="testin")
    testout = model.connect(testin)
    sess.track([
        testout,
    ])

    query_targets = []
    def error_wrapper(T, Y):
        inspector.add(Y, 'output')
        err = cross_entropy_loss(T, Y)
        query_targets.append(err)
        return err

    raw_inshape[0] = nbatch
    train_inshape = raw_inshape
    train_outshape = [nbatch, 10]
    train_input = tc.EVariable(train_inshape, label="trainin")
    train_output = tc.EVariable(train_outshape, label="trainout")
    normalized = train_input / 255. - 0.5
    train_err = tc.sgd_train(model, normalized, train_output,
        tc.get_adagrad(0.01), err_func=error_wrapper)
    sess.track([train_err])
    inspector.add(normalized, "normalized_input")
    tc.optimize(sess, "cfg/optimizations.json")

    qs = q.Statement(query_targets)
    conv_res = qs.find("""{"op": {
        "opname": "ADD",
        "args": [{"op": {
            "opname": "PERMUTE",
            "args": [{"op": {
                "opname": "CONV",
                "args": [
                    {"symb": ""},
                    {"op": {
                        "opname": "REVERSE",
                        "args": [{"leaf": {
                            "shape": [16, 3, 5, 5]
                        }}]
                    }}
                ]
            }}]
        }}]
    }}""")
    conv_res2 = qs.find("""{"op": {
        "opname": "ADD",
        "args": [{"op": {
            "opname": "PERMUTE",
            "args": [{"op": {
                "opname": "CONV",
                "args": [
                    {"symb": ""},
                    {"op": {
                        "opname": "REVERSE",
                        "args": [{"leaf": {
                            "shape": [20, 16, 5, 5]
                        }}]
                    }}
                ]
            }}]
        }}]
    }}""")
    conv_res3 = qs.find("""{"op": {
        "opname": "ADD",
        "args": [{"op": {
            "opname": "PERMUTE",
            "args": [{"op": {
                "opname": "CONV",
                "args": [
                    {"symb": ""},
                    {"op": {
                        "opname": "REVERSE",
                        "args": [{"leaf": {
                            "shape": [20, 20, 5, 5]
                        }}]
                    }}
                ]
            }}]
        }}]
    }}""")
    assert(
        len(conv_res) == 1 and
        len(conv_res2) == 1 and
        len(conv_res3) == 1)
    inspector.add(conv_res[0], "first_conv")
    inspector.add(conv_res2[0], "second_conv")
    inspector.add(conv_res3[0], "third_conv")

    # train
    for i, data in enumerate(cifar):
        labels = np.zeros((nbatch, 10))
        for j, label in enumerate(data['label']):
            labels[j][label] = 1
        print('expected label:\n{}'.format(labels))
        train_input.assign(data['image'].astype(np.float))
        train_output.assign(labels.astype(np.float))
        for j in range(nepochs):
            sess.update_target([train_err])
            trained_err = train_err.get()
            print('done epoch {}'.format(j))
            if j % show_every_n == show_every_n - 1:
                guess_err = trained_err.as_numpy()
                err = np.average(guess_err)
                print(('training {}th image, epoch {}\n'+
                    'training error:\n{}\n'+
                    'average training error:\n{}')
                    .format(i, j + 1, guess_err, err))

    # test
    tds = tfds.load('cifar10',
        split=tfds.Split.TEST,
        batch_size=1,
        shuffle_files=True)
    image = next(tfds.as_numpy(tds))
    names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    ]
    testin.assign(image['image'].astype(np.float))
    sess.update_target([testout])
    print('class: {}'.format(image['label']))
    print(testout.get())

    plt.imshow(image['image'].reshape(*raw_inshape[1:]))
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
