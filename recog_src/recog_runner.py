import time
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from recog_runtoolbox import fill_feed_dict, do_eval
from recog_model import placeholder_inputs, graph_model, calcul_loss, training, evaluation


class Predictor:
    def __init__(self,HYPARMS):
        self.init(HYPARMS)

    def predict(self, input):
        with self.sess.as_default():
            return self.classified.eval(feed_dict = {self.placebundle.x: input,
                                                    self.placebundle.keep_prob: 1})

    def init(self,HYPARMS):
        self.sess = tf.Session()
        self.placebundle = placeholder_inputs(1)

        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
        self.sess.run(self.init)

        self.logits = graph_model(self.placebundle)
        self.sftmax = tf.nn.softmax(self.logits)
        self.classified= tf.argmax(self.sftmax,1)

        ckpt = tf.train.get_checkpoint_state(HYPARMS.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print "Model loaded"
        else:
            print "No checkpoint file found"
