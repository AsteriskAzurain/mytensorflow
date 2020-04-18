import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread('0_3.png')    # 图片读取
img = cv2.resize(img, (64, 64))/255   # 图片尺寸压缩和归一化
img_new = np.float32(np.reshape(img, [1, 64, 64, 3]))   # 将图片shape改为4维
w1 = tf.random_normal([3, 3, 3, 32])  # filter

conv1 = tf.nn.conv2d(img_new, w1, strides=[1, 1, 1, 1], padding='SAME')  # 卷积操作
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 进行池化操作

sess = tf.Session()
conv = sess.run(conv1)
pool = sess.run(pool1)
sess.close()
cv2.imwrite('conv.jpg', conv[0, :, :, 15]*500)   # 将卷积结果的某一个面可视化呈现
cv2.imwrite('pool.jpg', pool[0, :, :, 15]*100)   # 将池化结果的某一个面可视化呈现


