import os.path
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST_data_folder='/Users/apple/tensorflow/MNIST/MNIST_data'
mnist = input_data.read_data_sets(MNIST_data_folder, one_hot = True)



# 输入，x为2维浮点张量,None用以指代batch的大小，意即 x 的数量不定
x = tf.placeholder("float", [None, 784])
# 输出，每行代表一个10维的one_hot向量，指代某个数字
y_ = tf.placeholder("float", [None, 10])

sess = tf.InteractiveSession()



# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 是用一个较小的正数来初始化偏置 项，以避免神经元节点输出恒为0的问题
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积和池化
# 卷积使用1步长（stride size），0边距（padding size）的模板
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

# 第一层卷积。卷积在每个5x5的patch中算出32个特征。
# 卷积的权重张量形状是 [5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的 通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#为了用这一层，我们把 x 变成一个4d向量，其第2、第3维对应图片的宽、高，
# 最后一维代表图片的颜色通道数(因 为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3
x_image = tf.reshape(x, [-1, 28, 28, 1])

#把x_image和权值向量卷积，加上偏置，使用relu激活函数，最后进行max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积。为了构建一个更深的网络，我们会把几个类似的层堆叠起来。
# 第二层中，每个5x5的patch会得到64个特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
# 图片尺寸减小到7x7(两次池化)
# 我们加入一个有1024个神经元的全连接层，用于处理整个图片。
# 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，我们在输出层之前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层使用softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

'''
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('mnist_logs', sess.graph)
total_step = 0
'''
# 训练和评估。用ADAM优化器来做梯度最速下降
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
for i in range(1000):

    batch = mnist.train.next_batch(50)

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob: 0.5})



print("test accuracy %g"%accuracy.eval(feed_dict={
x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
