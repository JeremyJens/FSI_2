import gzip
import cPickle
import sys

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
train_y = one_hot(train_y, 10) #El 10 es por el numero de digitos

train_x2, train_y2 = valid_set
train_y2 = one_hot(train_y2, 10)

train_x3, train_y3 = test_set
train_y3 = one_hot(train_y3, 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 4)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(4)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(4, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

#h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

#loss = tf.reduce_sum(tf.square(y_ - y))
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_size = 1000
last = sys.maxint
now = 0
epoch = 0
while (now > 0.00001*last +last or now < last - 0.00001*last) and epoch < 100:
    last = now
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    #print "VALIDACION"
    now = sess.run(loss, feed_dict={x: train_x2, y_: train_y2})
    print "Epoch #:", epoch, "Error: ", now
    epoch += 1

    #result = sess.run(y, feed_dict={x: train_x2})
    #for b, r in zip(train_y2, result):
    #    print b, "-->", r
    #print "----------------------------------------------------------------------------------"

print "TEST"

error = 0
result = sess.run(y, feed_dict={x: train_x3})
i = 1
for b, r in zip(train_y3 , result):
    #print i, "-->" ,b, "-->", r
    if np.argmax(b) != np.argmax(r):
		#print "-----> ERROR"
		error = error + 1
    i += 1

print "Errores = ",error