# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:32:17 2020

@author: Administrator
"""

import numpy as np
import tensorflow as tf

# 定义参数
num_words=2000  # 字典的最大长度
maxlen=80  # 句子的最大长度
batch_size = 128  # 批尺寸
num_classes = 46  # 独热编码输入的单位的数量

# 数据读取与预处理——对齐列数据并计算长度
# 通过tf.keras.datasets.reuters接口加载数据集，该数据集包含有11228条新闻，共分成46个主题
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(path='../reuters.npz', num_words=num_words)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen,padding='post')   # 预处理，设定最大长度，不足的补0,使数据对齐
x_test=tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen,padding='post')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # 打印长度
len_train = np.count_nonzero(x_train,axis=1)  # 计算每一行的非0数的长度
len_test = np.count_nonzero(x_test, axis=1)
print(len_train, len_test)
# 定义数据集并将不足一批次的剩余数据丢弃
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))   # 把训练集进行特征切片
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
res = next(iter(ds_train))  # 使用迭代器
# 深度学习一般使用float32，而numpy格式多为float64，所以需要转化
def preprocess(x,y):  # 数据处理函数
    x = tf.cast(x,dtype=tf.float32)/255
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=num_classes)
    return x,y
ds_train = ds_train.shuffle(1000).map(preprocess).batch(batch_size, drop_remainder = True)  # 每批次随机获取1000个点
ds_test = ds_test.shuffle(1000).map(preprocess).batch(batch_size, drop_remainder = True)
print(ds_train)
print(ds_test)


def mkMask(input_tensor,maxLen):
    '''
    计算变长RNN模型的掩码,根据序列长度生成对应的掩码
    :param input_tensor:输入标签文本序列的长度list，（batch_size）
    :param maxLen:输入标签文本的最大长度
    :return:
    '''
    shape_of_input = tf.shape(input_tensor)  # 输入序列的维度
    shape_of_output = tf.concat(axis=0,values=[shape_of_input,[maxLen]])  # 在第0个维度上连接
    oneDtensor = tf.reshape(input_tensor,shape=(-1,))  # 使用数组的reshape方法，可以创建一个改变了尺寸的新数组，原数组的shape保持不变
    flat_mask=tf.sequence_mask(oneDtensor,maxlen=maxLen)  #
    return tf.reshape(flat_mask,shape_of_output)

# 定义函数shared_routing_uhat，使用全连接网络，将RNN模型输出结果转换成动态路由中的uhat
def shared_routing_uhat(caps,out_caps_num,out_caps_dim):
    '''
    定义函数，将输入转化成uhat
    :param caps: 输入向量，[batch_size,max_len,cap_dims]
    :param out_caps_num:输出胶囊的个数
    :param out_cap_dim:输出胶囊的维度
    :return:
    '''
    batch_size,max_len = caps.shape[0],caps.shape[1]
    caps_uchat = tf.keras.layers.Dense(out_caps_num*out_caps_dim,activation='tanh')(caps)
    caps_uchat = tf.reshape(caps_uchat,[batch_size,max_len,out_caps_num,out_caps_dim])
    return caps_uchat

def _squash(in_caps,axes):
    '''
    定义_squash激活函数
    :param in_caps:
    :param axes:
    :return:
    '''
    _EPSILON = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(in_caps),axis=axes,keepdims=True)  # 计算张量tensor沿着某一维度的和，可以在求和后降维
    scalar_factor=vec_squared_norm/(1+vec_squared_norm)/tf.sqrt(vec_squared_norm+_EPSILON)  # 卷积
    vec_squared=scalar_factor*in_caps
    return vec_squared

# 定义函数masked_routing_iter,进行动态路由计算
# 在该函数的开始部分，对输入的序列长度进行掩码处理，使动态路由算法支持动态长度的序列数据输入
def masked_routing_iter(caps_uhat,seqLen,iter_num):
    '''
    动态路由计算
    :param caps_uhat:输入向量,(batch_size,max_len,out_caps_num,out_caps_dim)
    :param seqLen:
    :param iter_num:
    :return:
    '''
    assert iter_num>0
    # 获取批次和长度
    batch_size,max_len = tf.shape(caps_uhat)[0],tf.shape(caps_uhat)[1]
    # 获取胶囊的个数
    out_caps_num = int(tf.shape(caps_uhat)[2])
    seqLen = tf.where(tf.equal(seqLen,0),tf.ones_like(seqLen),seqLen)  # 将true位置元素替换为ａ中对应位置元素，false的替换为ｂ中对应位置元素
    mask = mkMask(seqLen,max_len)  # (batch_size,max_len)
    float_mask = tf.cast(tf.expand_dims(mask,axis=-1),dtype=tf.float32)  # (batch_size,max_len,1)
    # 初始化相似度权重b
    B = tf.zeros([batch_size,max_len,out_caps_num],dtype=tf.float32)  # 全部用0替代
    # 迭代更新相似度权重b
    for i in range(iter_num):
        # 计算相似度权重（耦合系数）c
        c=tf.keras.layers.Softmax(axis=2)(B)  # (batch_size,max_len,out_caps_num)
        c=tf.expand_dims(c*float_mask,axis=-1)  # (batch_size,max_len,out_caps_num,1)
        # 计算胶囊的输出(激活向量)v
        weighted_uhat=c*caps_uhat  # (batch_size,max_Len,out_caps_num, out_caps_dim)
        s=tf.reduce_sum(weighted_uhat,axis=1)  # (batch_size, out_caps_num, out_caps_dim)
        # squash非线性函数
        v=_squash(s,axes=[2])  # (batch_size, out_caps_num, out_caps_dim)
        v=tf.expand_dims(v,axis=1)  # (batch_size, 1, out_caps_num, out_caps_dim)
        # 更新相似度权重b
        B=tf.reduce_sum(caps_uhat*v,axis=-1)+B  # (batch_size, maxlen, out_caps_num)
    v_ret = tf.squeeze(v, axis=[1])  # shape(batch_size, out_caps_num, out_caps_dim)
    s_ret = s
    return v_ret, s_ret

# 定义函数routing_masked完成全部动态路由计算过程，对RNN模型的输出结果进行信息聚合
def routing_masked(in_x, xLen, out_caps_dim, out_caps_num, iter_num=3, dropout=None, is_train=False, scope=None):
    assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None
    b_sz = tf.shape(in_x)[0]
    with tf.variable_scope(scope or 'routing'):
        caps_uhat = shared_routing_uhat(in_x, out_caps_num, out_caps_dim, scope='rnn_caps_uhat')
        attn_ctx, S = masked_routing_iter(caps_uhat, xLen, iter_num)
        attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num*out_caps_dim])
        if dropout is not None:
            attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)  # 神经网络单元，按照一定的概率将其暂时从网络中丢弃，防止过拟合
    return attn_ctx
