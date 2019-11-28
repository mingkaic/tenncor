# source: https://github.com/erikvdplas/gru-rnn
import numpy as np

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

# Seed random
np.random.seed(0)

# Read data and setup maps for integer encoding and decoding.
data = open('models/data/gru_input.txt', 'r').read()
chars = sorted(list(set(data))) # Sort makes model predictable (if seeded).
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Activation functions
# NOTE: Derivatives are calculated using outcomes of their primitives (which are already calculated during forward prop).
def sigmoid(input, deriv=False):
    if deriv:
        return input*(1-input)
    else:
        return 1 / (1 + np.exp(-input))

def tanh(input, deriv=False):
    if deriv:
        return 1 - input ** 2
    else:
        return np.tanh(input)

# Derivative is directly calculated in backprop (in combination with cross-entropy loss function).
def softmax(input):
    # Subtraction of max value improves numerical stability.
    e_input = np.exp(input - np.max(input))
    return e_input / e_input.sum()

# Hyper parameters
h_size, o_size, N = vocab_size, vocab_size, vocab_size # Hidden size is set to vocab_size, assuming that level of abstractness is approximately proportional to vocab_size (but can be set to any other value).
seq_length = 25 # Longer sequence lengths allow for lengthier latent dependencies to be trained.
learning_rate = 1e-1

class OldModel:
    def __init__(self, h_size, o_size, N):
        # OldModel parameter initialization
        self.Wz = np.random.uniform(-0.05, 0.05, [h_size, N])
        self.Uz = np.random.uniform(-0.05, 0.05, [h_size, h_size])
        self.bz = np.zeros((h_size, 1))

        self.Wr = np.random.uniform(-0.05, 0.05, [h_size, N])
        self.Ur = np.random.uniform(-0.05, 0.05, [h_size, h_size])
        self.br = np.zeros((h_size, 1))

        self.Wh = np.random.uniform(-0.05, 0.05, [h_size, N])
        self.Uh = np.random.uniform(-0.05, 0.05, [h_size, h_size])
        self.bh = np.zeros((h_size, 1))

        self.Wy = np.random.uniform(-0.05, 0.05, [o_size, h_size])
        self.by = np.zeros((o_size, 1))

    def connect(self, inputs, hprev):
        x, z, r, h_hat = {}, {}, {}, {}
        h, y, p = {-1: hprev}, {}, {} # Dictionaries contain variables for each timestep.

        # Forward prop
        state = hprev
        for t in range(len(inputs)):
            # Set up one-hot encoded input
            xinput = np.zeros((vocab_size, 1))
            xinput[inputs[t]] = 1

            x[t] = xinput

            # Calculate update and reset gates
            update_gate = sigmoid(np.dot(self.Wz, xinput) + np.dot(self.Uz, state) + self.bz)
            reset_gate = sigmoid(np.dot(self.Wr, xinput) + np.dot(self.Ur, state) + self.br)

            z[t] = update_gate
            r[t] = reset_gate

            # Calculate hidden units'
            hidden_state = tanh(
                np.dot(self.Wh, xinput) +
                np.dot(self.Uh, reset_gate * state) + self.bh)
            state = update_gate * state + (1 - update_gate) * hidden_state

            h_hat[t] = hidden_state
            h[t] = state

            # Regular output unit
            y[t] = np.dot(self.Wy, state) + self.by # Dense

            # Probability distribution
            p[t] = softmax(y[t])

        return p, y, h, h_hat, r, x, z

class GRU:
    def __init__(self, h_size, N):
        self.ugate = rcn.Dense(h_size, eteq.Shape([N + h_size]),
            weight_init=rcn.unif_xavier_init(1),
            bias_init=rcn.zero_init(), label="update_gate")
        self.rgate = rcn.Dense(h_size, eteq.Shape([N + h_size]),
            weight_init=rcn.unif_xavier_init(1),
            bias_init=rcn.zero_init(), label="reset_gate")
        self.hgate = rcn.Dense(h_size, eteq.Shape([N + h_size]),
            weight_init=rcn.unif_xavier_init(1),
            bias_init=rcn.zero_init(), label="hidden_gate")

    def connect(self, inputs, init_state):
        state = init_state
        states = []
        inshape = inputs.shape()
        if len(inshape) < 2:
            nseq = 1
        else:
            nseq = inshape[0]
        for i in range(nseq):
            # Set up one-hot encoded input
            xinput = tc.slice(inputs, i, 1, 1)

            # Calculate update and reset gates
            xc = tc.concat(xinput, state, 0)
            update_gate = tc.sigmoid(self.ugate.connect(xc))
            reset_gate = tc.sigmoid(self.rgate.connect(xc))

            # Calculate hidden units
            xreset = tc.concat(xinput, reset_gate * state, 0)
            hidden_state = tc.tanh(self.hgate.connect(xreset))
            state = update_gate * state + (1 - update_gate) * hidden_state
            states.append(state)

        return tc.concat(states, 1), hidden_state

    def get_contents(self):
        return self.ugate.get_contents() +\
            self.rgate.get_contents() +\
            self.hgate.get_contents()

class SequentialModel:
    def __init__(self, h_size, o_size, N):
        self.gru = GRU(h_size, N)
        self.labeller = rcn.Dense(o_size, eteq.Shape([h_size]),
            weight_init=rcn.unif_xavier_init(1),
            bias_init=rcn.zero_init(), label="labeller")

    def connect(self, inputs, hprev):
        state, hnext = self.gru.connect(inputs, hprev)
        encoding = self.labeller.connect(state)
        return tc.softmax(encoding, 0, 1), hnext

    def get_contents(self):
        return self.gru.get_contents() +\
            self.labeller.get_contents()

def train(model, inputs, targets, hprev):
    # Initialize variables
    sequence_loss = 0
    # Forward prop
    p, y, h, h_hat, r, x, z = model.connect(inputs, hprev)

    for t in range(len(inputs)):
        # Cross-entropy loss
        loss = -np.log(p[t][targets[t]])
        sequence_loss += loss

    # Parameter gradient initialization
    dWy, dWh, dWr, dWz = np.zeros_like(model.Wy), np.zeros_like(model.Wh), np.zeros_like(model.Wr), np.zeros_like(model.Wz)
    dUh, dUr, dUz = np.zeros_like(model.Uh), np.zeros_like(model.Ur), np.zeros_like(model.Uz)
    dby, dbh, dbr, dbz = np.zeros_like(model.by), np.zeros_like(model.bh), np.zeros_like(model.br), np.zeros_like(model.bz)
    dhnext = np.zeros_like(h[0])

    # Backward prop
    for t in reversed(range(len(inputs))):
        # ∂loss/∂y
        dy = np.copy(p[t])
        dy[targets[t]] -= 1

        # ∂loss/∂Wy and ∂loss/∂by
        dWy += np.dot(dy, h[t].T)
        dby += dy

        # Intermediary derivatives
        dh = np.dot(model.Wy.T, dy) + dhnext
        dh_hat = np.multiply(dh, (1 - z[t]))
        dh_hat_l = dh_hat * tanh(h_hat[t], deriv=True)

        # ∂loss/∂Wh, ∂loss/∂Uh and ∂loss/∂bh
        dWh += np.dot(dh_hat_l, x[t].T)
        dUh += np.dot(dh_hat_l, np.multiply(r[t], h[t-1]).T)
        dbh += dh_hat_l

        # Intermediary derivatives
        drhp = np.dot(model.Uh.T, dh_hat_l)
        dr = np.multiply(drhp, h[t-1])
        dr_l = dr * sigmoid(r[t], deriv=True)

        # ∂loss/∂Wr, ∂loss/∂Ur and ∂loss/∂br
        dWr += np.dot(dr_l, x[t].T)
        dUr += np.dot(dr_l, h[t-1].T)
        dbr += dr_l

        # Intermediary derivatives
        dz = np.multiply(dh, h[t-1] - h_hat[t])
        dz_l = dz * sigmoid(z[t], deriv=True)

        # ∂loss/∂Wz, ∂loss/∂Uz and ∂loss/∂bz
        dWz += np.dot(dz_l, x[t].T)
        dUz += np.dot(dz_l, h[t-1].T)
        dbz += dz_l

        # All influences of previous layer to loss
        dh_fz_inner = np.dot(model.Uz.T, dz_l)
        dh_fz = np.multiply(dh, z[t])
        dh_fhh = np.multiply(drhp, r[t])
        dh_fr = np.dot(model.Ur.T, dr_l)

        # ∂loss/∂h𝑡₋₁
        dhnext = dh_fz_inner + dh_fz + dh_fhh + dh_fr

    return sequence_loss, dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz, h[-1]

def tc_train(model, sess, inputs, target_map, hprev, lr):
    prob, hnext = model.connect(inputs, hprev)
    # Cross-entropy loss
    target_probs = tc.reduce_sum(prob * target_map, 0, 1)
    loss = -tc.reduce_sum(tc.log(target_probs))

    contents = model.get_contents()
    vars = [
        contents[0], contents[1],
        contents[3], contents[4],
        contents[6], contents[7],
        contents[9], contents[10],
    ]
    grads = [(var, eteq.derive(loss, var)) for var in vars]

    # adagrad
    momentums = [(hprev, hnext)]
    group = []
    to_track = [hnext]
    momentum_update = [hnext]
    var_update = []
    for var, grad in grads:
        momentum = eteq.Variable(var.shape(), 0, "momentum_{}".format(str(var)))
        grad = tc.clip_by_range(grad, -5, 5)
        dmomentum = momentum + grad * grad
        dvar = var - lr * grad / tc.sqrt(momentum + 1e-8)
        momentums.append((momentum, dmomentum))
        group.append((var, dvar))
        to_track.append(dmomentum)
        to_track.append(dvar)
        momentum_update.append(dmomentum)
        var_update.append(dvar)
    update_groups = [(momentums, momentum_update), (group, var_update)]

    sess.track(to_track + [loss])

    def update():
        for group, to_update in update_groups:
            sess.update_target(to_update)
            for var, delta in group:
                var.assign(delta.get())

    return update, loss

def old_sample(model, h, seed_ix, n):
    # Initialize first word of old_sample ('seed') as one-hot encoded vector.
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = [seed_ix]

    for t in range(n):
        # Calculate update and reset gates
        z = sigmoid(np.dot(model.Wz, x) + np.dot(model.Uz, h) + model.bz)
        r = sigmoid(np.dot(model.Wr, x) + np.dot(model.Ur, h) + model.br)

        # Calculate hidden units
        h_hat = tanh(np.dot(model.Wh, x) + np.dot(model.Uh, np.multiply(r, h)) + model.bh)
        h = np.multiply(z, h) + np.multiply((1 - z), h_hat)

        # Regular output unit
        y = np.dot(model.Wy, h) + model.by

        # Probability distribution
        p = softmax(y)

        # Choose next char according to the distribution
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)

    return ixes

def sample(model, sess, sample_input, sample_prob, seed_ix, n):
    # Initialize first word of sample ('seed') as one-hot encoded vector.
    x = np.zeros((1, vocab_size))
    x[0][seed_ix] = 1
    sample_input.assign(x)
    ixes = [seed_ix]
    for _ in range(n):
        # Choose next char according to the distribution
        sess.update_target([sample_prob])
        pchoice = sample_prob.get().ravel()
        ix = np.random.choice(range(vocab_size), p=pchoice)

        x = np.zeros((1, vocab_size))
        x[0][ix] = 1
        sample_input.assign(x)

        ixes.append(ix)
    return ixes

def main():
    print_interval = 100

    oldModel = OldModel(h_size, o_size, N)
    mdWy, mdWh, mdWr, mdWz = np.zeros_like(oldModel.Wy), np.zeros_like(oldModel.Wh), np.zeros_like(oldModel.Wr), np.zeros_like(oldModel.Wz)
    mdUh, mdUr, mdUz = np.zeros_like(oldModel.Uh), np.zeros_like(oldModel.Ur), np.zeros_like(oldModel.Uz)
    mdby, mdbh, mdbr, mdbz = np.zeros_like(oldModel.by), np.zeros_like(oldModel.bh), np.zeros_like(oldModel.br), np.zeros_like(oldModel.bz)
    smooth_loss = -np.log(1.0/vocab_size)*seq_length
    p = 0
    for n in range(4000):
        # Reset memory if appropriate
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((h_size, 1))
            p = 0

        # Get input and target sequence
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        # Occasionally old_sample from oldModel and print result
        if n % print_interval == 0:
            old_sample_ix = old_sample(oldModel, hprev, inputs[0], 1000)
            txt = ''.join(ix_to_char[ix] for ix in old_sample_ix)
            print('----\n%s\n----' % (txt, ))

        # Get gradients for current oldModel based on input and target sequences
        loss, dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz, hprev = train(oldModel, inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Occasionally print loss information
        if n % print_interval == 0:
            print('iter %d, loss: %f, smooth loss: %f' % (n, loss, smooth_loss))

        # Update oldModel with adagrad (stochastic) gradient descent
        for param, dparam, mem in zip(
                [oldModel.Wy,  oldModel.Wh,  oldModel.Wr,  oldModel.Wz,  oldModel.Uh,  oldModel.Ur,  oldModel.Uz,  oldModel.by,  oldModel.bh,  oldModel.br,  oldModel.bz],
                [dWy, dWh, dWr, dWz, dUh, dUr, dUz, dby, dbh, dbr, dbz],
                [mdWy,mdWh,mdWr,mdWz,mdUh,mdUr,mdUz,mdby,mdbh,mdbr,mdbz]):
            np.clip(dparam, -5, 5, out=dparam)
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # Small added term for numerical stability

        # Prepare for next iteration
        p += seq_length


    sample_input = eteq.Variable([vocab_size], 0, 'sample_input')
    hidden_prev = eteq.Variable([h_size], 0, 'hidden_prev')
    target_map = eteq.Variable([seq_length, vocab_size], 0, 'target_map')
    sess = eteq.Session()

    model = SequentialModel(h_size, o_size, N)
    sample_prob, _ = model.connect(sample_input, hidden_prev)
    sess.track([sample_prob])

    xinput = eteq.Variable([seq_length, vocab_size], 0, 'xinput')
    trainer, tc_loss = tc_train(model, sess, xinput, target_map, hidden_prev, learning_rate)
    smooth_loss = -np.log(1.0/vocab_size)*seq_length

    p = 0
    for n in range(4000):
        # Reset memory if appropriate
        if p + seq_length + 1 >= len(data) or n == 0:
            hidden_prev.assign(np.zeros(h_size))
            p = 0

        # Get input and target sequence
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
        tmap = np.zeros([seq_length, vocab_size])
        for i, t in enumerate(targets):
            tmap[i][t] = 1

        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        imap = np.zeros((seq_length, vocab_size))
        for i, inp in enumerate(inputs):
            imap[i][inp] = 1

        target_map.assign(np.array(tmap))
        xinput.assign(np.array(imap))

        trainer()
        sess.update_target([tc_loss])
        smooth_loss = smooth_loss * 0.999 + tc_loss.get() * 0.001

        # Occasionally old_sample from oldModel and print result
        if n % print_interval == 0:
            sample_ix = sample(model, sess, sample_input, sample_prob, inputs[0], 1000)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n%s\n----' % (txt, ))

        # Occasionally print loss information
        if n % print_interval == 0:
            print('iter %d, loss: %f, smooth loss: %f' % (n, tc_loss.get(), smooth_loss))

        # Prepare for next iteration
        p += seq_length

if __name__ == "__main__":
    main()
