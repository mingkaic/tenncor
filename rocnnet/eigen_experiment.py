import tensorflow as tf
import numpy as np

# batch, height, width, in
shape = [3, 3, 3, 2]
data = np.array(range(np.prod(np.array(shape))),
	dtype=np.float32).reshape(shape) + 1

# height, width, in, out
kshape = [2, 2, 2, 4]
kdata = np.array(range(np.prod(np.array(kshape))),
	dtype=np.float32).reshape(kshape) + 1

image = tf.Variable(data)
kernel = tf.Variable(kdata)

conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], 'VALID')

sess = tf.Session()
sess.run(image.initializer)
sess.run(kernel.initializer)

print('image')
print(sess.run(image))
print('kernel')
print(sess.run(kernel))
print('conv')
print(sess.run(conv))
