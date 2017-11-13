import tensorflow as tf
import tensorflow.contrib.layers as layers

def multiheaded(input_data, num_actions, scope, reuse=False):
    K = 10
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("convolutional"):
            out = layers.convolution2d(input_data, num_outputs=32, kernel_size=8,
                                       stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4,
                                       stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3,
                                       stride=1, activation_fn=tf.nn.relu)
        flattened = layers.flatten(out)
        with tf.variable_scope("fully_connected"):
            heads = [layers.fully_connected(flattened, num_outputs=512,
                                             activation_fn=tf.nn.relu)
                     for _ in range(K)]
            heads = [layers.fully_connected(head_out, num_outputs=num_actions,
                                            activation_fn=tf.nn.relu)
                     for head_out in heads]
    return heads
