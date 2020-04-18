import tensorflow as tf

# 生成正态分布的随机数 1*2 1*5
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')  # name可以与model一起保存到文件中,后期载入模型时方便引入(会有默认值,但不好找)
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')  # w1 w2只是保存在内存中 不会保存到文件

saver = tf.train.Saver()   # 模型保存的类
sess = tf.Session()  # 启动会话
sess.run(tf.global_variables_initializer())  # 变量初始化
saver.save(sess, 'tf_learn/savemymodel/temp/modelnew')   # 保存模型 不写路径即保存到当前目录
'''
checkpoint: 日志
model.data-00000-of-00001: 模型中所有参数的取值
model.index: 所有变量的索引
model.meta: 计算图
'''
sess.close()


