import numpy


def convert_data_to_numpy(images, labels):
    images = convert_images_to_numpy(images)
    labels = convert_labels_to_numpy(labels)
    return images, labels

def convert_labels_to_numpy(labels):
    np_labels = numpy.array(labels, dtype=int)

    return np_labels

def convert_images_to_numpy(images):
    arr = numpy.empty([len(images), 28, 28], dtype=float) # todo 28,48 -> flexible
    for idx, image in enumerate(images):
        arr[idx][:][:] = convert_image_to_numpy(image)
    return arr

def convert_image_to_numpy(image):
    np_image = numpy.array(image, dtype=float)
    np_image = normalize_if_its_integer(np_image)
    return np_image

def normalize_if_its_integer(image):
    if (image > 1).any() :
        image = image / 255
    return image