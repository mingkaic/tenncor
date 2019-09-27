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
    def __init__(self, dbn, nbatch,
        pretrain_lr=0.1,
        train_lr=0.1,
        cdk=1):

        layers = dbn.get_layers()
        assert len(layers) > 2

        self.rbm_layers = layers[:-2]
        self.n_layers = len(self.rbm_layers)

        self.log_layer = rcn.SequentialModel("training")
        self.log_layer.add(layers[-2])
        self.log_layer.add(layers[-1])

        # setups:
        # general rbm sampling
        self.sample_outs = []
        trainx = eteq.scalar_variable(0, [dbn.get_ninput(), nbatch], "trainx")
        xin = trainx
        for rbm in self.rbm_layers:
            xin = sample_h_given_v(rbm, xin)
            self.sample_outs.append(xin)
        self.sample_ins = [trainx] + self.sample_outs[:-1]

        # layer-wise rbm reconstruction
        self.pretrain_sess = eteq.Session()
        self.rassigns = []
        self.rcosts = []

        for rbm, sample_in, sample_out in zip(self.rbm_layers, self.sample_ins, self.sample_outs):

            w, hb, _, vb, _ = rbm.get_contents()

            chain_it = sample_out
            for _ in range(cdk-1):
                chain_it = tc.random.rand_binom_one(rbm.connect(
                    sample_v_given_h(rbm, chain_it)))
            nv_samples = sample_v_given_h(rbm, chain_it)
            nh_means = rbm.connect(nv_samples)

            self.rassigns.append((
                w + pretrain_lr * (tc.matmul(tc.transpose(sample_in), sample_out) - tc.matmul(tc.transpose(nv_samples), nh_means)),
                vb + pretrain_lr * tc.reduce_mean_1d(sample_in - nv_samples, 1),
                hb + pretrain_lr * tc.reduce_mean_1d(sample_out - nh_means, 1),
            ))
            vhv = rbm.backward_connect(rbm.connect(sample_in))
            reconstruction_cost = -tc.reduce_mean(
                tc.reduce_sum_1d(sample_in * tc.log(vhv) + (1 - sample_in) * tc.log(1 - vhv), 0))
            self.rcosts.append(reconstruction_cost)

        self.pretrain_sess.track([e for tup in self.rassigns for e in tup] + self.rcosts)

        # logistic layer training

    def pretrain(self, x, sess, lr=0.1, k=1, epochs=100):
        # varx = self.sample_ins[0]
        # varx.assign(x.reshape(1, -1))
        # for i, (rbm, sample_in, sample_out, assigns, rcost) in enumerate(zip(
        #     self.rbm_layers, self.sample_ins, self.sample_outs, self.rassigns, self.rcosts)):

        #     w, hb, _, vb, _ = rbm.get_contents()

        #     for epoch in range(epochs):
        #         self.pretrain_sess.update_target(assigns, ignored=[sample_in])
        #         w.make_var().assign(assigns[0].get())
        #         vb.make_var().assign(assigns[1].get())
        #         hb.make_var().assign(assigns[2].get())

        #         if epoch % 100 == 0:
        #             # reconstruction error
        #             sess.update_target([rcost], ignored=[varx])
        #             eprint('Pre-training layer {}, epoch {}, cost {}'.format(i, epoch, rcost.get()))

        # self.pretrain_sess.update_target([sample_out], ignored=[sample_in])

        # pre-train layer-wise
        rxs = [eteq.variable(x)]
        assigns = []
        rcosts = []
        for i in range(self.n_layers):
            # train rbm layers (reconstruction) setup
            rbm = self.rbm_layers[i]
            rx = rxs[i]
            ry = sample_h_given_v(rbm, rx)
            rxs.append(ry)

            w, hb, _, vb, _ = rbm.get_contents()
            nh_samples = ry
            for step in range(k):
                nv_samples = sample_v_given_h(rbm, nh_samples)
                nh_means = rbm.connect(nv_samples)
                if step < k-1:
                    nh_samples = tc.random.rand_binom_one(nh_means)
            assigns.append((
                w + lr * (tc.matmul(tc.transpose(rx), ry) - tc.matmul(tc.transpose(nv_samples), nh_means)),
                vb + lr * tc.reduce_mean_1d(rx - nv_samples, 1),
                hb + lr * tc.reduce_mean_1d(ry - nh_means, 1),
            ))

            vhv = rbm.backward_connect(rbm.connect(rx))
            rcosts.append(-tc.reduce_mean(
                tc.reduce_sum_1d(rx * tc.log(vhv) + (1 - rx) * tc.log(1 - vhv), 0)))

        sess.track(rxs + rcosts + [e for ass in assigns for e in ass])

        for i in range(self.n_layers):
            # train rbm layers (reconstruction) setup
            rbm = self.rbm_layers[i]
            rx = rxs[i]
            ry = rxs[i + 1]

            w, hb, _, vb, _ = rbm.get_contents()
            dw, dvb, dhb = assigns[i]

            rcost = rcosts[i]

            for epoch in range(epochs):
                # train rbm layers (reconstruction)
                sess.update_target([dw, dvb, dhb], ignored=[rx])
                w.make_var().assign(dw.get())
                vb.make_var().assign(dvb.get())
                hb.make_var().assign(dhb.get())

                if epoch % 100 == 0:
                    # reconstruction error
                    sess.update_target([rcost], ignored=[rx])
                    eprint('Pre-training layer {}, epoch {}, cost {}'.format(i, epoch, rcost.get()))

            sess.update_target([ry], ignored=[rx])

    def finetune(self, x, y, sess, lr=0.1, L2_reg=0.0, epochs=100):
        # train log layer setup
        x_var = eteq.variable(x)
        y_var = eteq.variable(y)

        ignored = []
        for rbm in self.rbm_layers:
            x_var = sample_h_given_v(rbm, x_var)
            ignored.append(x_var)
        sess.track([x_var])
        sess.update_target([x_var])

        w, b, _ = tuple(self.log_layer.get_contents())
        p_y = self.log_layer.connect(x_var)
        d_y = y_var - p_y
        d_w = tc.matmul(tc.transpose(x_var), d_y)
        l2_regularized = d_w - L2_reg * w
        assigns = [
            w + lr * l2_regularized,
            b + lr * tc.reduce_mean_1d(d_y, 1)
        ]
        cost = -tc.reduce_mean(
            tc.reduce_sum_1d(y_var * tc.log(p_y) + (1 - y_var) * tc.log(1 - p_y), 0))

        sess.track(assigns + [cost])

        # train log_layer
        epoch = 0
        while epoch < epochs:

            # train log layer
            sess.update_target(assigns, ignored=ignored)

            w.make_var().assign(assigns[0].get())
            b.make_var().assign(assigns[1].get())

            if epoch % 100 == 0:
                # log layer error
                sess.update_target([cost], ignored=ignored)
                finetune_cost = cost.get()

                eprint('Training epoch {}, cost is {}'.format(epoch, finetune_cost))

            lr *= 0.95
            epoch += 1

pretrain_lr=0.1
pretraining_epochs=1000
k=1
finetune_lr=0.1
finetune_epochs=200

trainingx = np.array([
    [1,1,1,0,0,0],
    [1,0,1,0,0,0],
    [1,1,1,0,0,0],
    [0,0,1,1,1,0],
    [0,0,1,1,0,0],
    [0,0,1,1,1,0]])
y = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1]])

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

trainer = DBNTrainer(dbn, trainingx.shape[1],
    pretrain_lr=pretrain_lr,
    train_lr=finetune_lr,
    cdk=1)

# pre-training (TrainUnsupervisedDBN)
pretraining_sess = eteq.Session()
trainer.pretrain(trainingx, pretraining_sess, lr=pretrain_lr, k=1, epochs=pretraining_epochs)

# fine-tuning (DBNSupervisedFineTuning)
training_sess = eteq.Session()
trainer.finetune(trainingx, y, training_sess, lr=finetune_lr, epochs=finetune_epochs)

# test
x = np.array([1, 1, 0, 0, 0, 0])
sess = eteq.Session()
var = eteq.variable(x)
out = dbn.connect(var)
sess.track([out])
sess.update_target([out])
# since x is similar to first 3 rows of trainingx, expect results simlar to first 3 rows of y [1, 0]
print(out.get())
