from __future__ import print_function
import sys
import numpy as np

import json
import decimal

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, decimal.Decimal):
            return {
                'value': str(o),
                'type': str(type(o).__name__) # you can use another code to provide the type.
            }
        return super(CustomEncoder, self).default(o)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

record_file = 'rocnnet/demo/data/dbn_record.txt'

try:
    with open(record_file) as f:
        super_records = json.loads(f.read())
except:
    super_records = None

i = 0
def load_record():
    global i
    if super_records:
        out = super_records[i]
        i += 1
    else:
        out = None
    return out

class RNGWrapper:
    def __init__(self, rng, record):
        self.rng = rng
        self.record_mode = record is None
        if record is None:
            self.record = []
        else:
            self.record = record

    def pop_records(self):
        return self.record.pop(0)

    def binomial(self, size, n, p, ctx = ''):
        inputs = (list(size), n, p.tolist())
        if self.record_mode:
            output = self.rng.binomial(size=size, n=n, p=p)
            self.record.append((ctx, inputs, output))
        else:
            ctx, rec_inputs, output = self.pop_records()
            rec_inputs = tuple(rec_inputs)
            assert inputs == rec_inputs
        return np.array(output)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(x):
    e = np.exp(x)
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


def sample_h_given_v(rbm, x, rng):
    return tc.random.rand_binom_one(rbm.connect(x))


class DBN(object):
    def __init__(self, x=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2):

        self.real_rbm = []

        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)

        assert self.n_layers > 0

        self.np_rng = RNGWrapper(np.random.RandomState(123), record=load_record())

        # construct multi-layer
        input_size = n_ins
        data = [
            [
                [0.06548972853262056, -0.07128688834987351, -0.09104951547859896],
                [0.01710492302763042, 0.07315632326185437, -0.025631179958513023],
                [0.16025473279487182, 0.0616099128616211, -0.006356032838546355],
                [-0.03596082726861649, -0.052273994616376857, 0.07634990246134721],
                [-0.020475918440125196, -0.1467740344634772, -0.03398524822318952],
                [0.07933180191067857, -0.10583608984883333, -0.10818274795083582]],
            [
                [-0.049099128691461214, 0.2622594420780899, 0.2961066788025864],
                [0.001224450589557724, 0.08263530119474077, -0.2562544032804695],
                [-0.12180967878645271, -0.056782525364245495, 0.24420610525557723]],
        ]

        for i in range(self.n_layers):
            d = np.array(data[i])

            # construct rbm_layer
            rbm_layer = RBM(w=d, n_visible=input_size, n_hidden=hidden_layer_sizes[i])
            self.rbm_layers.append(rbm_layer)

            real_rbm = rcn.RBM(hidden_layer_sizes[i], input_size,
                weight_init=lambda shape, label: eteq.variable(d, label),
                bias_init=rcn.zero_init(), label="{}".format(i))
            self.real_rbm.append(real_rbm)

            input_size = hidden_layer_sizes[i - 1]

        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(n_in=hidden_layer_sizes[-1], n_out=n_outs)

        self.real_log_layer = rcn.Dense(n_outs, hidden_layer_sizes[-1],
            weight_init=rcn.zero_init(),
            bias_init=rcn.zero_init(), label="log_layer")
        final_layer = rcn.SequentialModel("final")
        final_layer.add(self.real_log_layer)
        final_layer.add(rcn.softmax(0))
        self.real_rbm.append(final_layer)

        v_mean = self.rbm_layers[0].propup(x)
        self.final_input = self.np_rng.binomial(size=v_mean.shape, n=1, p=v_mean, ctx='final_layer_create')
        self.rfinal_input = sample_h_given_v(self.real_rbm[0], eteq.variable(x), self.np_rng)

    def pretrain(self, x, lr=0.1, k=1, epochs=100):
        # pre-train layer-wise
        layer_input = x
        rlayer_input = eteq.variable(x)
        for i in range(self.n_layers):
            if i > 0:
                rlayer_input = sample_h_given_v(self.real_rbm[i - 1], rlayer_input, self.np_rng)

                v_mean = self.rbm_layers[i-1].propup(layer_input)
                layer_input = self.np_rng.binomial(size=v_mean.shape, n=1, p=v_mean, ctx='pretrain')

            rbm = self.rbm_layers[i]
            for epoch in range(epochs):

                v_mean = rbm.propup(layer_input)
                ph_sample = rbm.np_rng.binomial(size=v_mean.shape, n=1, p=v_mean, ctx='sample_h_given_v')

                chain_start = ph_sample
                for step in range(k):
                    if step == 0:
                        _, nv_samples, nh_means, nh_samples = rbm.gibbs_hvh(chain_start)
                    else:
                        _, nv_samples, nh_means, nh_samples = rbm.gibbs_hvh(nh_samples)

                rbm.W += lr * (np.dot(layer_input.T, ph_sample)
                                - np.dot(nv_samples.T, nh_means))
                rbm.vbias += lr * np.mean(layer_input - nv_samples, axis=0)
                rbm.hbias += lr * np.mean(ph_sample - nh_means, axis=0)


                if epoch % 100 == 0:
                    cost = rbm.get_reconstruction_cross_entropy(layer_input)
                    eprint('Pre-training layer {}, epoch {}, cost {}'.format(i, epoch, cost))

    def finetune(self, y, lr=0.1, epochs=100):
        v_mean = self.rbm_layers[-1].propup(self.final_input)
        layer_input = self.np_rng.binomial(size=v_mean.shape, n=1, p=v_mean, ctx='finetune')

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(x=layer_input, y=y, lr=lr)
            if epoch % 100 == 0:
                self.finetune_cost = self.log_layer.negative_log_likelihood(layer_input, y)
                eprint('Training epoch {}, cost is {}'.format(epoch, self.finetune_cost))

            lr *= 0.95
            epoch += 1

    def predict(self, x):
        layer_input = x

        for i in range(self.n_layers):
            layer_input = self.rbm_layers[i].propup(layer_input)

        out = self.log_layer.log_predict(layer_input)
        return out

class RBM(object):
    def __init__(self, w, n_visible=2, n_hidden=3):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        a = 1. / n_visible
        W = w
        hbias = np.zeros(n_hidden)  # initialize h bias 0
        vbias = np.zeros(n_visible)  # initialize v bias 0

        self.np_rng = RNGWrapper(np.random.RandomState(1234), record=load_record())
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.np_rng.binomial(
            size=h1_mean.shape,   # discrete: binomial
            n=1, p=h1_mean, ctx='sample_h_given_v')

        return [h1_mean, h1_sample]


    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.np_rng.binomial(
            size=v1_mean.shape,   # discrete: binomial
            n=1, p=v1_mean, ctx='sample_v_given_h')

        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = np.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = np.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]


    def get_reconstruction_cross_entropy(self, x):
        pre_sigmoid_activation_h = np.dot(x, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =  - np.mean(
            np.sum(x * np.log(sigmoid_activation_v) +
            (1 - x) * np.log(1 - sigmoid_activation_v),
                      axis=1))

        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(np.dot(h, self.W.T) + self.vbias)
        return reconstructed_v


class LogisticRegression(object):
    def __init__(self, n_in, n_out):
        self.W = np.zeros((n_in, n_out))  # initialize W 0
        self.b = np.zeros(n_out)          # initialize bias 0

    def train(self, x, y, lr=0.1, L2_reg=0.00):
        p_y_given_x = softmax(np.dot(x, self.W) + self.b)
        d_y = y - p_y_given_x

        self.W += lr * np.dot(x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * np.mean(d_y, axis=0)

    def negative_log_likelihood(self, x, y):
        sigmoid_activation = softmax(np.dot(x, self.W) + self.b)

        cross_entropy = - np.mean(
            np.sum(y * np.log(sigmoid_activation) +
            (1 - y) * np.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy

    def log_predict(self, x):
        return softmax(np.dot(x, self.W) + self.b)



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
    [0,0,1,1,1,0]])
y = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1]])

# construct DBN
dbn = DBN(x=x, n_ins=6, hidden_layer_sizes=[3, 3], n_outs=2)

# pre-training (TrainUnsupervisedDBN)
dbn.pretrain(x, lr=pretrain_lr, k=1, epochs=pretraining_epochs)

# fine-tuning (DBNSupervisedFineTuning)
dbn.finetune(y, lr=finetune_lr, epochs=finetune_epochs)

# test
x = np.array([1, 1, 0, 0, 0, 0])
print(dbn.predict(x))

if dbn.np_rng.record:
    records = [dbn.np_rng.record]
else:
    records = []
for layer in dbn.rbm_layers:
    if layer.np_rng.record:
        records.append(layer.np_rng.record)

if len(records) > 0:
    with open(record_file, 'w') as f:
        f.write(json.dumps(records, cls=CustomEncoder))
