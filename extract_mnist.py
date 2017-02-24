import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

from PIL import Image
import numpy as np
n = 10
for i in range(n):
    img = trX[i].reshape([28,28])
    img = img *255
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    filename = 'mnist_idx_'+ str(i) + '_label_' +str(np.argmax(trY[i]))+'.bmp'
    print(np.argmax(trY[i]))
    im.save(filename)
    im.show()
