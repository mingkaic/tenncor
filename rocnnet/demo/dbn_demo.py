import sys
import time
import argparse

import numpy as np

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

prog_description = 'Demo dbn_trainer'

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
        eteq.seed(args.seedval)
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
    model = rcn.SequentialModel("demo")
    model.add(rcn.RBM(3, 6,
        weight_init=rcn.unif_xavier_init(),
        bias_init=rcn.zero_init(), label="0"))
    model.add(rcn.RBM(3, 3,
        weight_init=rcn.unif_xavier_init(),
        bias_init=rcn.zero_init(), label="1"))
    model.add(rcn.Dense(2, eteq.Shape([3]),
        weight_init=rcn.zero_init(),
        bias_init=rcn.zero_init(), label="log_layer"))
    model.add(rcn.softmax(0))

    untrained = model.clone()
    try:
        print('loading ' + args.load)
        trained = rcn.load_file_seqmodel(args.load, "demo")
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))
        trained = model.clone()

    trainer = rcn.DBNTrainer(model, x.shape[0],
        pretrain_lr = 0.1,
        train_lr = 0.1,
        cdk = args.cdk)

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
    sess = eteq.Session()
    var = eteq.variable(x)
    untrained_out = untrained.connect(var)
    out = model.connect(var)
    trained_out = trained.connect(var)
    sess.track([untrained_out, out, trained_out])
    sess.update_target([untrained_out, out, trained_out])
    # since x is similar to first 3 rows of x, expect results simlar to first 3 rows of y [1, 0]
    print('untrained_out: ', untrained_out.get())
    print('out: ', out.get())
    print('trained_out: ', trained_out.get())

    try:
        print('saving')
        if model.save_file(args.save):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if '__main__' == __name__:
    main(sys.argv[1:])
