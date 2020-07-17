import tensorflow as tf

# return true if left < right
def version_lt(left, right):
    left = left.split('.')
    right = right.split('.')
    for lv, rv in zip(left, right):
        lv = int(lv)
        rv = int(rv)
        if lv < rv:
            return True
        elif lv > rv:
            return False
    return False

def tf_init():
    if version_lt(tf.__version__, '2.0.0'):
        tf_sess = tf.Session
    else:
        tf_sess = tf.compat.v1.Session
        tf.compat.v1.disable_v2_behavior()

    return tf_sess
