
class Hyparms():
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.1
        self.max_steps = 5000
        self.log_dir = 'logs'
        self.input_data_dir = 'data'
        self.ckpt_dir = 'trained'
        self.ckpt_name = 'trained_weight'
        self.dropout_rate = 0.9
        self.fake_data = False

class Weight:
    def __init__(self,
                 W_conv1,
                 W_conv2,
                 W_fc1,
                 W_fc2):
        self.W_conv1 = W_conv1
        self.W_conv2 = W_conv2
        self.W_fc1 = W_fc1
        self.W_fc2 = W_fc2

class Placebundle:
    def __init__(self,
                 x,
                 y_,
                 W,
                 B,
                 keep_prob):
        self.x = x
        self.y_ = y_
        self.W = W
        self.B = B
        self.keep_prob = keep_prob