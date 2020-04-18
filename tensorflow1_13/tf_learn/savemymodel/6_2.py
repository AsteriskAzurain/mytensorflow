import tensorflow as tf

sess = tf.Session()
saver = tf.train.import_meta_graph('tf_learn/savemymodel/temp/modelnew.meta')  # 导入保存好的计算图
saver.restore(sess, 'tf_learn/savemymodel/temp/modelnew')  # 导入计算图中的所有参数 不导入会导致模型里的参数未初始化
# output: INFO:tensorflow:Restoring parameters from tf_learn/savemymodel/temp/modelnew
w1_new = tf.get_default_graph().get_tensor_by_name('w1:0')   # 通过name获取保存好的tensor “:0”是序号 指文件中第一个出现的w1
'''
In[20]:  w1_new
Out[20]: <tf.Tensor 'w1:0' shape=(2,) dtype=float32_ref>
'''
sess.run(w1_new)
sess.close()

