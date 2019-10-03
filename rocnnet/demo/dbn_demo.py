import numpy as np
import time

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

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

eteq.seed(int(time.time()))

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

trainer = rcn.DBNTrainer(dbn, x.shape[0],
    pretrain_lr=pretrain_lr,
    train_lr=finetune_lr,
    cdk=k)

def pretrain_log(epoch, layer):
    if epoch % 100 == 0:
        print('Pre-training layer {}, epoch {}, cost {}'.format(
            layer, epoch, trainer.reconstruction_cost(layer)))

# pre-training (TrainUnsupervisedDBN)
trainer.pretrain(x, nepochs=pretraining_epochs, logger=pretrain_log)

def finetune_log(epoch):
    if epoch % 100 == 0:
        print('Training epoch {}, cost {}'.format(
            epoch, trainer.training_cost()))

# fine-tuning (DBNSupervisedFineTuning)
trainer.finetune(x, y, nepochs=finetune_epochs, logger=finetune_log)

# test
x = np.array([1, 1, 0, 0, 0, 0])
sess = eteq.Session()
var = eteq.variable(x)
out = dbn.connect(var)
sess.track([out])
sess.update_target([out])
# since x is similar to first 3 rows of x, expect results simlar to first 3 rows of y [1, 0]
print(out.get())
