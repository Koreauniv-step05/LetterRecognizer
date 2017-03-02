import numpy
from DataConverter.object.dataset import DataSet


def shuffle_numpy_together(nparr1, nparr2):
    assert nparr1.shape[0] == nparr2.shape[0], (
            'nparr1.shape: %s nparr1.shape: %s' % (nparr1.shape, nparr2.shape))

    perm = numpy.arange(nparr1.shape[0])
    numpy.random.shuffle(perm)
    return nparr1[perm], nparr2[perm]

def shuffle_dataset_together(dataset):
    images, labels = shuffle_numpy_together(dataset.images, dataset.labels)
    return DataSet(images, labels)