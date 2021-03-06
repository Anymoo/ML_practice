#导入Minst数据集
from tensorflow.examples.tutorials.mnist import input_data
MNIST_data_folder='/Users/apple/PycharmProjects/ML_practice/tensorflow/MNIST_data'
mnist=input_data.read_data_sets(MNIST_data_folder,one_hot=True)
print (mnist.train.next_batch(1))


#导入tensorflow库
import tensorflow as tf

#输入变量，把28*28的图片变成一维数组（丢失结构信息）
x = tf.placeholder("float",[None,784])

#权重矩阵，把28*28=784的一维输入，变成0-9这10个数字的输出
w = tf.Variable(tf.zeros([784,10]))
#偏置
b = tf.Variable(tf.zeros([10]))

#核心运算，其实就是softmax（x*w+b）
y = tf.nn.softmax(tf.matmul(x,w) + b)

#这个是训练集的正确结果
y_ = tf.placeholder("float",[None,10])

#交叉熵，作为损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#梯度下降算法，最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化，在run之前必须进行的
init = tf.initialize_all_variables()
#创建session以便运算
sess = tf.Session()
sess.run(init)

#迭代1000次
for i in range(1000):
  #获取训练数据集的图片输入和正确表示数字
  batch_xs, batch_ys = mnist.train.next_batch(100)
  #运行刚才建立的梯度下降算法，x赋值为图片输入，y_赋值为正确的表示数字
  sess.run(train_step,feed_dict = {x:batch_xs, y_: batch_ys})

#tf.argmax获取最大值的索引。比较运算后的结果和本身结果是否相同。
#这步的结果应该是[1,1,1,1,1,1,1,1,0,1...........1,1,0,1]这种形式。
#1代表正确，0代表错误
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#tf.cast先将数据转换成float，防止求平均不准确。
#tf.reduce_mean由于只有一个参数，就是上面那个数组的平均值。
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
#输出
print (sess.run(accuracy,feed_dict={x:mnist.test.images,y_: mnist.test.labels}))
