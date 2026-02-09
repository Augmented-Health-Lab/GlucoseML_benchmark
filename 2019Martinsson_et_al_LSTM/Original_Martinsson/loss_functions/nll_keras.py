import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
# import tensorflow_probability as tfp

def load():
    return tf_nll

def tf_nll(y_true, y_pred):
    y_var = y_pred[:,:1]
    y_mean = y_pred[:,1:]
    #y_std  = tf.sqrt(y_var)
    y_std  = y_var

    dist = tf.distributions.Normal(loc=y_mean, scale=y_std)
    probs = dist.prob(y_true)
    return tf.reduce_mean(-tf.log(tf.keras.backend.epsilon() + probs))
