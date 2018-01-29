from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path
import sys
import time

from six.moves import xrange
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', '/Users/apple/PycharmProjects/ML_practice/MNITST/MNIST_data',
                    'Directory to put the training data.')
flags.DEFINE_string('log_dir', '/Users/apple/PycharmProjects/ML_practice/MNITST/log_dir',
                    'Directory to put the log data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

'''填充反馈字典
函数会查询给定的 DataSet ，
索要下一批次 batch_size 的图像和标签，
与占位符相匹配的Tensor则会包含下一批次的图像和标签
目的：在训练时对应次数自动填充字典
输入：数据源data_set，来自input_data.read_data_sets
     图片holder:images_pl,来自placeholder_inputs()
     标签holder:labels_pl,来自placeholder_inputs()
输出：反馈字典feed_dict.
'''
def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                    FLAGS.fake_data)
    # 以占位符为哈希键，创建一个Python字典对象，键值则是其代表的反馈Tensor
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    # 记录预测正确的数目
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f'%
            (num_examples, true_count, precision))

def run_training():
    data_set = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
    # 默认在Graph下运行
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)
        logits = mnist.inference(images_placeholder,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2)
        loss = mnist.loss(logits, labels_placeholder)
        train_op = mnist.training(loss, FLAGS.learning_rate)
        eval_correct = mnist.evaluation(logits, labels_placeholder)
        # 汇总tensor
        summary = tf.summary.merge_all()
        # 建立初始化机制
        init = tf.global_variables_initializer()
        # 建立保存机制
        saver = tf.train.Saver()
        #建立session
        sess = tf.Session()
        # 建立一个SummaryWriter输出汇聚的tensor
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(init)
        # 开始训练
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # 获得当前循环次数
            feed_dict = fill_feed_dict(data_set.train,
                                       images_placeholder,
                                       labels_placeholder)
            '''sess.run() 会返回一个有两个元素的元组。其中每一个 Tensor 对象，
            对应了返回的元组 中的numpy数组，而这些数组中包含了当前这步训练中对应Tensor的值。
            由于 train_op 并不会产生输出，其在返 回的元祖中的对应元素就是 None ，
            所以会被抛弃。但是，如果模型在训练中出现偏差， loss Tensor的值可能 会变成NaN，
            所以我们要获取它的值，并记录下来'''
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)'% (step,
                                        loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            # 每1000次测试模型
            if (step + 1) % 1000 == 0 or (step + 1 ) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                print('Traning data eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_set.train)
                print('Validation data eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_set.validation)
                print('test data eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_set.test)

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
