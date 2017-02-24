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
    saver = tf.train.Saver()

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(HYPARMS.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model loaded"
        else:
            print "No checkpoint file found"

        feed_dict = fill_feed_dict(input,
                                   placebundle.x,
                                   placebundle.y_,
                                   placebundle.keep_prob,
                                   HYPARMS)

        return classified.eval(feed_dict)