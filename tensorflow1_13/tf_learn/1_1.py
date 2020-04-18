import tensorflow as tf

# 第一步：定义计算（计算图）
a = tf.constant([1.0, 2])
b = tf.constant([2.0, 3])
res = a+b
print(res)
# output: Tensor("add:0", shape=(2,), dtype=float32)
# ↑ res的结构信息

# 第二步：执行计算（会话中）
sess = tf.Session()    # 启动会话
res1 = sess.run(res)
sess.close()   # 关闭会话

print(res1)
# output: [3. 5.]
# ↑ 真正的运算结果
