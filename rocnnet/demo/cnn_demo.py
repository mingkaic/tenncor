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

nbatch = 4
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0001

# batch, height, width, in
train_inshape = [nbatch, 32, 32, 3]
train_outshape = [nbatch, 10]

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

model.add(rcn.Dense(10, 320,
    weight_init=rcn.unif_xavier_init(),
    bias_init=rcn.zero_init(), label="fc")) # outputs [nbatch, 10]
model.add(rcn.softmax(0))

sess = eteq.Session()
train_input = eteq.Variable(train_inshape)
train_output = eteq.Variable(train_outshape)
train = rcn.sgd_train(model, sess,
    train_input, train_output, rcn.get_sgd(0.9))

test_inshape = [1, 32, 32, 3]

testin = eteq.Variable(test_inshape, label="testin")
testout = model.connect(testin)
sess.track([
    testout,
])
sess.optimize("cfg/optimizations.rules")

# train

def preprocess(image, label):
    pass

# train_input.assign(image_batch)
# train_output.assign(label_batch)
# train()

# test
print(testout.shape())
