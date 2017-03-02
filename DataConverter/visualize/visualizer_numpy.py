
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from DataConverter.load.classifier import shuffle_dataset_together


def show_random_nbyn_data(dataset, n=5):
    show_nbyn_data(shuffle_dataset_together(dataset))

def show_nbyn_data(dataset, n=5, idx_range=None):
    if idx_range is None:
        idx_range = range(n ** 2)

    for idx in idx_range:
        plt.subplot(n,n,idx+1)
        show_nth_data(dataset,idx)

    plt.show()

def show_nth_data(dataset,idx):
    show_numpy_image(dataset.images[idx], dataset.labels[idx])

def show_nbyn_images(images, n=5, idx_range=None):
    if idx_range is None:
        idx_range = range(n ** 2)

    for idx in idx_range:
        plt.subplot(n,n,idx+1)
        show_nth_image(images,idx)
    plt.show()

def show_nth_image(images,idx=0):
    show_numpy_image(images[idx])

def show_numpy_image(image, title=None):
    if image.ndim is 1:
        image = image.reshape(28,28) # todo 4828
    if title is not None:
        plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])