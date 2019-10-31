# source: https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html (warning: disable javascript)
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

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

import tensorflow_datasets as tfds
import numpy as np

import matplotlib.pyplot as plt

cifar_name = 'cifar10'
assert cifar_name in tfds.list_builders()

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
model = rcn.SequentialModel("demo")
model.add(rcn.Conv([5, 5], 3, 16, "0",
    zero_padding=[2, 2])) # outputs [nbatch, 32, 32, 16]
model.add(rcn.relu())
model.add(rcn.maxpool2d([1, 2])) # outputs [nbatch, 16, 16, 16]
model.add(rcn.Conv([5, 5], 16, 20, "1",
    zero_padding=[2, 2])) # outputs [nbatch, 16, 16, 20]
model.add(rcn.relu())
model.add(rcn.maxpool2d([1, 2])) # outputs [nbatch, 8, 8, 20]
model.add(rcn.Conv([5, 5], 20, 20, "2",
    zero_padding=[2, 2])) # outputs [nbatch, 8, 8, 20]
model.add(rcn.relu())
model.add(rcn.maxpool2d([1, 2])) # outputs [nbatch, 4, 4, 20]

model.add(rcn.Dense(10, eteq.Shape([4, 4, 20]), # weight has shape [10, 4, 4, 20]
    weight_init=rcn.unif_xavier_init(),
    bias_init=rcn.zero_init(), params=eteq.constant([1, 1, 2, 2, 3, 3]), label="fc")) # outputs [nbatch, 10]
model.add(rcn.softmax(1))

sess = eteq.Session()

raw_inshape[0] = 1
test_inshape = raw_inshape
testin = eteq.Variable(test_inshape, label="testin")
testout = model.connect(testin)
sess.track([
    testout,
])

raw_inshape[0] = nbatch
train_inshape = raw_inshape
train_outshape = [nbatch, 10]
train_input = eteq.Variable(train_inshape, label="trainin")
train_output = eteq.Variable(train_outshape, label="trainout")
normalized = train_input / 255 - 0.5
train = rcn.sgd_train(model, sess,
    normalized, train_output, rcn.get_sgd(0.5))

sess.optimize("cfg/optimizations.rules")

# train
for i, data in enumerate(cifar):
    labels = np.zeros((nbatch, 10))
    for j, label in enumerate(data['label']):
        labels[j][label] = 1
    print('expected label:\n{}'.format(labels))
    train_input.assign(data['image'].astype(np.float))
    train_output.assign(labels.astype(np.float))
    for j in range(nepochs):
        trained_err = train()
        print('done epoch {}'.format(j))
        if j % show_every_n == show_every_n - 1:
            guess_err = trained_err.as_numpy()
            err = np.max(guess_err, axis=1)
            print(('training {}th image, epoch {}\n'+
                'training error:\n{}\n'+
                'training error max:\n{}')
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
