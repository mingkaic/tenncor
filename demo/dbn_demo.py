import sys
import time
import argparse

import numpy as np

import tenncor as tc

prog_description = 'Demo dbn model'

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
    parser.add_argument('--pre_nepoch', dest='pretrain_epochs', type=int, nargs='?', default=1000,
        help='Number of epochs when pretraining (default: 1000)')
    parser.add_argument('--nepochs', dest='finetune_epochs', type=int, nargs='?', default=200,
        help='Number of epochs when finetuning (default: 200)')
    parser.add_argument('--cdk', dest='cdk', type=int, nargs='?', default=1,
        help='Length of the Contrastive divergence chain (default: 1)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='models/dbn.onnx',
        help='Filename to load pretrained model (default: models/dbn.onnx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        tc.seed(args.seedval)
        np.random.seed(args.seedval)

    x = np.array([
        [1,1,1,0,0,0],
        [1,0,1,0,0,0],
        [1,1,1,0,0,0],
        [0,0,1,1,1,0],
        [0,0,1,1,0,0],
        [0,0,1,1,1,0],
        [0,0,1,1,0,1]])
    y = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 0]])

    # construct DBN
    rbms = [
        tc.api.layer.rbm(6, 3),
        tc.api.layer.rbm(3, 3),
    ]
    dense = tc.api.layer.dense([3], [2],
        kernel_init=tc.api.init.zeros())
    softmax_dim = 0

    rbm_interlace = zip([rbm.fwd() for rbm in rbms], len(rbms) * [tc.api.layer.bind(tc.api.sigmoid)])
    model = tc.api.layer.link([e for inters in rbm_interlace for e in inters] +
        [dense, tc.api.layer.bind(lambda x: tc.api.softmax(x, softmax_dim, 1))])
    untrained = model.deep_clone()
    trained = model.deep_clone()
    try:
        print('loading ' + args.load)
        trained = tc.load_from_file(args.load)[0]
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))

    trainer = tc.DBNTrainer(rbms, dense, softmax_dim, x.shape[0],
        pretrain_lr = 0.1, train_lr = 0.1, cdk = args.cdk)

    def pretrain_log(epoch, layer):
        if epoch % 100 == 0:
            print('Pre-training layer {}, epoch {}, cost {}'.format(
                layer, epoch, trainer.reconstruction_cost(layer)))

    def finetune_log(epoch):
        if epoch % 100 == 0:
            print('Training epoch {}, cost {}'.format(
                epoch, trainer.training_cost()))

    # pre-training (TrainUnsupervisedDBN)
    trainer.pretrain(x, nepochs=args.pretrain_epochs, logger=pretrain_log)

    # fine-tuning (DBNSupervisedFineTuning)
    trainer.finetune(x, y, nepochs=args.finetune_epochs, logger=finetune_log)

    # test
    x = np.array([1, 1, 0, 0, 0, 0])
    var = tc.variable(x)
    untrained_out = untrained.connect(var)
    out = model.connect(var)
    trained_out = trained.connect(var)

    tc.optimize("cfg/optimizations.json")

    # since x is similar to first 3 rows of x, expect results simlar to first 3 rows of y [1, 0]
    print('untrained_out: ', untrained_out.get())
    print('out: ', out.get())
    print('trained_out: ', trained_out.get())

    try:
        print('saving')
        if tc.save_to_file(args.save, [model]):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if '__main__' == __name__:
    main(sys.argv[1:])
