from train_src.train_class import Hyparms
from train_src.train_runner import run_training
import tensorflow as tf

def load_params():
    HYPARMS = Hyparms()

    return HYPARMS

def main(_):
    HYPARMS = load_params()
    if tf.gfile.Exists(HYPARMS.log_dir):
        tf.gfile.DeleteRecursively(HYPARMS.log_dir)
    tf.gfile.MakeDirs(HYPARMS.log_dir)
    run_training(HYPARMS)

if __name__ == '__main__':
  tf.app.run(main=main)