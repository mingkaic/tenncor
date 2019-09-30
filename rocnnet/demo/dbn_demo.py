from __future__ import print_function
import sys
import numpy as np

from dbg.eteq_mocker import custom_unary

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def sample_h_given_v(rbm, x):
    return tc.random.rand_binom_one(rbm.connect(x))

def sample_v_given_h(rbm, x):
    return tc.random.rand_binom_one(rbm.backward_connect(x))

class DBNTrainer(object):
    def __init__(self, model, batch_size,
        pretrain_lr=0.1,
        train_lr=0.1,
        cdk=1,
        L2_reg=0.0,
        lr_scaling=0.95):

        layers = model.get_layers()
        assert len(layers) > 2

        self.input_size = model.get_ninput()
        self.output_size = model.get_noutput()
        self.batch_size = batch_size
        self.trainx = eteq.scalar_variable(0, [batch_size, model.get_ninput()], "trainx")
        self.trainy = eteq.scalar_variable(0, [batch_size, model.get_noutput()], 'trainy')
        self.nlayers = len(layers) - 2

        self.pretrain_sess = eteq.Session()
        self.train_sess = eteq.Session()
        self.rupdates = []
        self.rcosts = []

        rbm_layers = layers[:-2]
        dense_layer = layers[-2]
        softmax_layer = layers[-1]

        # setups:
        # general rbm sampling
        self.sample_pipes = [self.trainx]
        for i in range(self.nlayers):
            self.sample_pipes.append(sample_h_given_v(
                rbm_layers[i], self.sample_pipes[i]))

        # layer-wise rbm reconstruction
        to_track = []
        for i in range(self.nlayers):
            rbm = rbm_layers[i]
            rx = self.sample_pipes[i]
            ry = self.sample_pipes[i + 1]

            w, hb, _, vb, _ = rbm.get_contents()
            chain_it = ry
            for _ in range(cdk-1):
                chain_it = tc.random.rand_binom_one(rbm.connect(
                    sample_v_given_h(rbm, chain_it)))
            nv_samples = sample_v_given_h(rbm, chain_it)
            nh_means = rbm.connect(nv_samples)
            dwleft = tc.matmul(tc.transpose(rx), ry)
            dwright = tc.matmul(tc.transpose(nv_samples), nh_means)
            dw = w + pretrain_lr * (
                dwleft - dwright)
            dhb = hb + pretrain_lr * \
                tc.reduce_mean_1d(ry - nh_means, 1)
            dvb = vb + pretrain_lr * \
                tc.reduce_mean_1d(rx - nv_samples, 1)
            self.rupdates.append((
                (w, dw),
                (hb, dhb),
                (vb, dvb),
            ))

            vhv = rbm.backward_connect(rbm.connect(rx))
            rcost = -tc.reduce_mean(
                tc.reduce_sum_1d(rx * tc.log(vhv) +
                    (1 - rx) * tc.log(1 - vhv), 0))
            self.rcosts.append(rcost)

            to_track.append(dw)
            to_track.append(dhb)
            to_track.append(dvb)
            to_track.append(rcost)

        to_track.append(self.sample_pipes[-1])
        self.pretrain_sess.track(to_track)

        # logistic layer training
        # todo: improve this adhoc way of training log layer
        w, b = tuple(dense_layer.get_contents())
        final_out = softmax_layer.connect(dense_layer.connect(self.sample_pipes[-1]))
        diff = self.trainy - final_out
        l2_regularized = tc.matmul(tc.transpose(
            self.sample_pipes[-1]), diff) - \
            L2_reg * w

        wshape = w.shape()[::-1]
        bshape = b.shape()[::-1]
        tlr_placeholder = eteq.scalar_variable(train_lr, [], 'learning_rate')
        dw = w + tc.extend(tlr_placeholder, 0, wshape) * l2_regularized
        db = b + tc.extend(tlr_placeholder, 0, bshape) * tc.reduce_mean_1d(diff, 1)
        dtrain_lr = tlr_placeholder * lr_scaling

        self.tupdates = (
            (w, dw),
            (b, db),
            (tlr_placeholder, dtrain_lr)
        )
        self.tcost = -tc.reduce_mean(
            tc.reduce_sum_1d(self.trainy * tc.log(final_out) +
                (1 - self.trainy) * tc.log(1 - final_out), 0))

        self.train_sess.track([
            self.sample_pipes[-1],
            dw,
            db,
            dtrain_lr,
            self.tcost
        ])

    def pretrain(self, x, nepochs=100):
        self.trainx.assign(x)
        for i in range(self.nlayers):
            # train rbm layers (reconstruction) setup
            to_ignore = [self.sample_pipes[i]]
            (w, dw), (hb, dhb), (vb, dvb) = self.rupdates[i]

            rcost = self.rcosts[i]

            for epoch in range(nepochs):
                # train rbm layers (reconstruction)
                self.pretrain_sess.update_target([dw, dvb, dhb], ignored=to_ignore)
                w.make_var().assign(dw.get())
                vb.make_var().assign(dvb.get())
                hb.make_var().assign(dhb.get())

                if epoch % 100 == 0:
                    # reconstruction error
                    self.pretrain_sess.update_target([rcost], ignored=to_ignore)
                    eprint('Pre-training layer {}, epoch {}, cost {}'.format(i, epoch, rcost.get()))

            if i < self.nlayers - 1:
                self.pretrain_sess.update_target([self.sample_pipes[i + 1]], ignored=to_ignore)

    def finetune(self, x, y, nepochs=100):
        # train log layer setup
        self.trainx.assign(x)
        self.trainy.assign(y)
        (w, dw), (b, db), (lr, dlr) = self.tupdates

        # assert len(self.sample_pipes) > 1, since self.nlayers > 0
        to_ignore = self.sample_pipes[-1]
        self.train_sess.update_target([to_ignore], ignored=[self.sample_pipes[-2]])

        for epoch in range(nepochs):
            # train log layer
            self.train_sess.update_target([dw, db, dlr], ignored=[to_ignore])
            print(db.get())

            w.make_var().assign(dw.get())
            b.make_var().assign(db.get())
            lr.make_var().assign(dlr.get())

            if epoch % 100 == 0:
                # log layer error
                self.train_sess.update_target([self.tcost], ignored=[to_ignore])
                eprint('Training epoch {}, cost is {}'.format(epoch, self.tcost.get()))

pretrain_lr=0.1
pretraining_epochs=1000
k=1
finetune_lr=0.1
finetune_epochs=200

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

eteq.seed(0)

# construct DBN
dbn = rcn.SequentialModel("dbn")
dbn.add(rcn.RBM(3, 6,
    weight_init=rcn.unif_xavier_init(),
    bias_init=rcn.zero_init(), label="0"))
dbn.add(rcn.RBM(3, 3,
    weight_init=rcn.unif_xavier_init(),
    bias_init=rcn.zero_init(), label="1"))
dbn.add(rcn.Dense(2, 3,
    weight_init=rcn.zero_init(),
    bias_init=rcn.zero_init(), label="log_layer"))
dbn.add(rcn.softmax(0))

if len(sys.argv) > 1:
    trainer = rcn.DBNTrainer(dbn, x.shape[0],
        pretrain_lr=pretrain_lr,
        train_lr=finetune_lr,
        cdk=k)
else:
    trainer = DBNTrainer(dbn, x.shape[0],
        pretrain_lr=pretrain_lr,
        train_lr=finetune_lr,
        cdk=k)

# pre-training (TrainUnsupervisedDBN)
trainer.pretrain(x, nepochs=pretraining_epochs)

# fine-tuning (DBNSupervisedFineTuning)
trainer.finetune(x, y, nepochs=finetune_epochs)

# test
x = np.array([1, 1, 0, 0, 0, 0])
sess = eteq.Session()
var = eteq.variable(x)
out = dbn.connect(var)
sess.track([out])
sess.update_target([out])
# since x is similar to first 3 rows of x, expect results simlar to first 3 rows of y [1, 0]
print(out.get())
