from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# 数据集有10类，0-9
NUM_CLASSES = 10

# 图像尺寸28*28
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_PIXELS

def inference(images, hidden1_units, hidden2_units):
    '''参数含义：
    images：图片占位符；h1:第一隐层的大小；
    '''
    # hidden1
    with tf.name_scope('hidden1_units'):
        # 每一层所使用的权重和偏差都在tf.Variable实例中生成，并且包含了各自期望的shape
        '''通过 tf.truncated_normal 函数初始化权重变量，给赋予的shape则是一个二维tensor，
        其中第一个维度代表该层中权重变量所连接（connect from）的单元数量，
        第二个维度代表该层中权重变量所 连接到的（connect to）单元数量'''
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units])), name='biases')
        # ReLu(Rectified Linear Units)激活函数
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # hidden2
    with tf.name_scope('hidden2_units'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units)),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
        hidden2 =  tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
                                name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES], name='biases'))
        logits = tf.matmul(hidden2_units, weights) + biases
    return logits

# 程序会返回包含了损失值的Tensor
def loss(logits, labels):
    labels = tf.to_int64(labels)
    # labels 为one-hot形式，logits必须未经缩放，该操作内部会对logits使用softmax操作
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    # 从loss中获取tensor，记录loss标量
    tf.summary.scalar('loss', loss)
    # 按照所要求的学习效率（learning rate）应用梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # 生成一个变量用于保存全局训练步骤（global training step）的数值
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 使用minimize()函数更新系统中的三角权重（triangle weights）、增加全局步骤的操作
    # 操作被称为 train_op ，是Ten sorFlow会话（session）诱发一个完整训练步骤所必须运行的操作
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
