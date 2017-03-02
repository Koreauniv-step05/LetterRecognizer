from recog_src.recog_class import Hyparms
from recog_src.recog_runner import Predictor
from recog_src.recog_loadimage import load_allpath, load_image
from recog_src.recog_preproc import preprocess
import tensorflow as tf
from DataConverter.visualize.visualizer_numpy import show_numpy_image

def load_params():
    HYPARMS = Hyparms()

    return HYPARMS

def letter_recognizer(path, visualize=False):
    HYPARMS = load_params()
    predictor = Predictor(HYPARMS)

    image = load_image(path)
    image = preprocess(image)
    print(path)
    pred = predictor.predict(image)
    if visualize:
        show_numpy_image(image[0], pred)
        import matplotlib
        matplotlib.pyplot.show()
    return pred