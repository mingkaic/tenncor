from memory_profiler import profile
import tensorflow as tf
import time

shape = [100, 1000, 1000]

@profile
def main():
    with tf.compat.v1.Session() as sess:
        a = tf.compat.v1.get_variable("W", shape, initializer=tf.random_uniform_initializer(1., 2.), dtype=tf.float32, trainable=False)
        b = tf.compat.v1.get_variable("W2", shape, initializer=tf.random_uniform_initializer(1., 2.), dtype=tf.float32, trainable=False)
        c = tf.matmul(tf.matmul(a,b),a)

        #sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(a.initializer)
        sess.run(b.initializer)
        stuff = sess.run([c])

        print('hello')

if __name__ == '__main__':
    main()
