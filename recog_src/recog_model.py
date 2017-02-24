import tensorflow as tf
from recog_class import Placebundle, Weight
from recog_toolbox import weight_variable, bias_variable, max_pool_2x2, conv2d

def placeholder_inputs(batch_size):
    w_conv1 = weight_variable([5, 5, 1, 32])
    w_conv2 = weight_variable([5, 5, 32, 64])
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    w_fc2 = weight_variable([1024, 10])
    W = Weight(w_conv1,w_conv2,w_fc1,w_fc2)

    b_conv1 = bias_variable([32])
    b_conv2 = bias_variable([64])
    b_fc1 = bias_variable([1024])
    b_fc2 = bias_variable([10])
    B = Weight(b_conv1, b_conv2, b_fc1, b_fc2)

    keep_prob = tf.placeholder(tf.float32)

    x = tf.placeholder(tf.float32, shape=[batch_size, 784])
    y_ = tf.placeholder(tf.int32, shape=[batch_size])

    placebundle = Placebundle(x, y_, W,B,keep_prob)

    return placebundle

def graph_model(placebundle):
    x = placebundle.x
    W_conv1, W_conv2 = placebundle.W.W_conv1, placebundle.W.W_conv2
    b_conv1, b_conv2 = placebundle.B.W_conv1, placebundle.B.W_conv2
    W_fc1, W_fc2 = placebundle.W.W_fc1, placebundle.W.W_fc2
    b_fc1, b_fc2 = placebundle.B.W_fc1, placebundle.B.W_fc2
    keep_prob = placebundle.keep_prob


    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return logits

def calcul_loss(logits, placebundle):
    y_ = placebundle.y_

    labels = tf.to_int64(y_)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss

def training(loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, placebundle):
    y_ = placebundle.y_

    correct = tf.nn.in_top_k(logits, y_, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
