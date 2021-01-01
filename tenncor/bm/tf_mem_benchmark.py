from memory_profiler import memory_usage
import tensorflow as tf
import time

shape = [1000, 1000, 1000]
 
#@profile
def main():
    with tf.compat.v1.Session() as sess:
        a = tf.compat.v1.get_variable("W", shape, initializer=tf.random_uniform_initializer(1., 2.), dtype=tf.float32, trainable=False)
        b = tf.compat.v1.get_variable("W2", shape, initializer=tf.random_uniform_initializer(1., 2.), dtype=tf.float32, trainable=False)
        c = (a + b) * a

        #sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(a.initializer)
        sess.run(b.initializer)
        stuff = sess.run([c])

        print('hello')

if __name__ == '__main__':
    main()
    #mem_usage = memory_usage(main)
    #print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    #print('Maximum memory usage: %s' % max(mem_usage))
