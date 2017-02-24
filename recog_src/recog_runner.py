import time
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from recog_runtoolbox import fill_feed_dict, do_eval
from recog_model import placeholder_inputs, graph_model, calcul_loss, training, evaluation

def run_recognize(input, HYPARMS):

    placebundle = placeholder_inputs(1)
    logits = graph_model(placebundle)
    sftmax = tf.nn.softmax(logits)
    classified = tf.argmax(sftmax,1)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        start_time = time.time()

        feed_dict = fill_feed_dict(input,
                                   placebundle.x,
                                   placebundle.y_,
                                   placebundle.keep_prob,
                                   HYPARMS)

        return classified.eval(feed_dict)