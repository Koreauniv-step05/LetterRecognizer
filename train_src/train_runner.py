import time
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from train_runtoolbox import fill_feed_dict, do_eval
from train_model import placeholder_inputs, graph_model, calcul_loss, training, evaluation

def run_training(HYPARMS):
    data_sets = input_data.read_data_sets(HYPARMS.input_data_dir, HYPARMS.fake_data)

    with tf.Graph().as_default():
        placebundle = placeholder_inputs(HYPARMS.batch_size)
        logits = graph_model(placebundle)
        loss = calcul_loss(logits, placebundle)
        train_op = training(loss, HYPARMS.learning_rate)
        eval_correct = evaluation(logits, placebundle)

        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            saver = tf.train.Saver()

            for step in xrange(HYPARMS.max_steps):
                start_time = time.time()

                feed_dict = fill_feed_dict(data_sets.train,
                                           placebundle.x,
                                           placebundle.y_,
                                           placebundle.keep_prob,
                                           HYPARMS)

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)

                duration = time.time() - start_time
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.
                    # summary_str = sess.run(summary, feed_dict=feed_dict)
                    # summary_writer.add_summary(summary_str, step)
                    # summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 1000 == 0 or (step + 1) == HYPARMS.max_steps:
                    checkpoint_file = os.path.join(HYPARMS.ckpt_dir, HYPARMS.ckpt_name)
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            placebundle.x,
                            placebundle.y_,
                            placebundle.keep_prob,
                            data_sets.train,
                            HYPARMS)
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            placebundle.x,
                            placebundle.y_,
                            placebundle.keep_prob,
                            data_sets.validation,
                            HYPARMS)
                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    do_eval(sess,
                            eval_correct,
                            placebundle.x,
                            placebundle.y_,
                            placebundle.keep_prob,
                            data_sets.test,
                            HYPARMS)