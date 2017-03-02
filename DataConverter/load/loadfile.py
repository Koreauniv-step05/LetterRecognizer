from PIL import Image
from loaddir import load_allpath,get_folder_name


def load_data(root):
    paths = load_allpath(root)
    return load_allimage(paths), load_alllabel(paths)


def load_image(path):
    return Image.open(path)

def load_allimage(paths):
    res = []

    for path in paths:
        res.append(load_image(path))

    return res

def load_label(path, map):
    return map[get_folder_name(path)]

def load_alllabel(paths):
    res = []
    map = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
    }

    for path in paths:
        res.append(load_label(path, map))

    return res