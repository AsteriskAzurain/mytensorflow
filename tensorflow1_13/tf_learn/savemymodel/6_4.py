import tensorflow as tf
# pip install opencv-python
import cv2
import numpy as np

path = 'tf_learn/savemymodel/testimages/'

tf.reset_default_graph()  # 调用model前重置计算图
sess = tf.Session()
saver = tf.train.import_meta_graph('tf_learn/savemymodel/model/softmax_model_new.meta')
saver.restore(sess, 'tf_learn/savemymodel/model/softmax_model_new')
graph = tf.get_default_graph()  # 获取当前计算图
input = graph.get_tensor_by_name('input:0')  # 模型输入节点
output = graph.get_tensor_by_name('output:0')  # 模型输出节点

for i in range(30):
    img = cv2.imread(path + str(i) + '.jpg')[:, :, 0] / 255  # 读取图片数据 [行,列,颜色通道(黑白)] 归一化: ÷255
    # shape: 28*28 → 1*784 便于模型处理
    img = img.reshape([1, 28 * 28])  # 进行维度转化
    pre = sess.run(output, feed_dict={input: img})  # 将新样本放入模型中进行预测
    '''
    图片7pre:
    [[3.9466759e-06 2.1888924e-07 6.2070598e-05 2.6356074e-06 3.4874749e-05
      3.6486899e-06 9.9985373e-01 1.8240222e-08 2.8810671e-05 9.9781655e-06]]
    '''
    res = np.argmax(pre, 1)  # 预测标签
    # 图片7res:[6]
    print('图片 ', str(i) + '.jpg 中的数字是: ', res[0])
sess.close()
# output: 图片  0.jpg 中的数字是:  3
