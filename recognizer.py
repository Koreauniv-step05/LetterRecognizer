from recog_src.recog_class import Hyparms
from recog_src.recog_runner import Predictor
from recog_src.recog_loadimage import load_allpath, load_image
from recog_src.recog_preproc import preprocess
import tensorflow as tf

def load_params():
    HYPARMS = Hyparms()

    return HYPARMS

def main(_):
    HYPARMS = load_params()
    if tf.gfile.Exists(HYPARMS.log_dir):
        tf.gfile.DeleteRecursively(HYPARMS.log_dir)
    tf.gfile.MakeDirs(HYPARMS.log_dir)

    predictor = Predictor(HYPARMS)

    paths = load_allpath(HYPARMS.recog_data_dir)
    for path in paths:
        image = load_image(path)
        image = preprocess(image)
        print(path)
        print(predictor.predict(image))

if __name__ == '__main__':
  tf.app.run(main=main)