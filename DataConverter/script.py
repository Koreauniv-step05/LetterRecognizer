from load.loadfile import load_data
from load.converter import convert_data_to_numpy
from process.normalizer_pillow import normalize_images
from visualize.visualizer_numpy import show_nbyn_images, show_random_nbyn_data
from DataConverter import inputdata


rootdir = "data"
datasets = inputdata.read_data_sets(rootdir)
show_random_nbyn_data(datasets.train)
#show_nbyn_images(datasets.test.images)