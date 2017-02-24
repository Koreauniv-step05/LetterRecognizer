from PIL import Image
import os

def load_allpath(path):
    res = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(os.path.abspath(path), file)
            res.append(filepath)
    return res

def load_image(path):
    return Image.open(path)