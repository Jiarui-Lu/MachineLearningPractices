import tensorflow as tf
import numpy as np


def linear_regression(real_B,real_A):



    real_A = np.reshape(real_A,(-1,1))
    real_B = np.reshape(real_B, (-1, 1))

    x_ = tf.placeholder(tf.float32, [None, 1])
    y_ = tf.placeholder(tf.float32, [None, 1])  # y_为测试集结果数据

    weight = tf.Variable(tf.ones([1, 1]))
    bias = tf.Variable(tf.ones([1]))

    y = tf.matmul(x_, weight) + bias

    loss = tf.reduce_mean(tf.square(y - y_)) #批量线性回归用这个损失函数
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    init = tf.global_variables_initializer()

    flag = True
    with tf.Session() as sess:
        sess.run(init)

        count = 0
        loss_temp = 0
        while flag:

            feed = {x_: real_A, y_: real_B}
            sess.run(train_step, feed_dict=feed)

            loss_res = sess.run(loss, feed_dict=feed)

            if loss_temp == loss_res:
                flag = False
            count += 1
            loss_temp = loss_res

        weight = sess.run(weight)
        bias = sess.run(bias)

        return weight[0][0],bias[0]

if __name__ == "__main__":

    x = np.arange(1,100,1)
    y = 3 * x + 1
    for _ in range(10):
        weight, bias = linear_regression(y,x)
        print(weight)
        print(bias)


