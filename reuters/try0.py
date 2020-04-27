import tensorflow as tf

# data preparation
num_epochs = 200
batch_size = 128
learning_rate = 0.001
num_classes = 46
total_words = 40000
max_news_words = 80
embedding_len = 200

# 解决ValueError: Object arrays cannot be loaded when allow_pickle=False
# 原因: numpy版本的问题,numpy的的版本和keras没有完全兼容
# old = np.load
# np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

# 通过tf.keras.datasets.reuters接口加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=total_words)

# 每句保留80个词 不足的补0
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_news_words)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_news_words)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# output:(8982, 80) (8982,) (2246, 80) (2246,)

# 将标签y转为one hot编码
# y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes, 'int64')
# y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes, 'int64')

# 定义数据集
# 将样本数据按照指定批次制作成tf.data.Dataset接口的数据集
# 并将不足一批次的剩余数据丢弃
# 批次(batch_size)设为128
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# ↑ tf.data.Dataset.from_tensor_slices
# ↑ 该函数是dataset核心函数之一，它的作用是把给定的元组、列表和张量等数据进行特征切片
# ↑ 元素逐个转换为Tensor对象然后依次放入Dataset中


def preprocess(x, y):  # 将标签 y 进行了 one hot 编码
    y = tf.one_hot(y, depth=num_classes)
    return x, y


# batch 处理数据的时候让 drop_remainder = True, 这样可以丢弃掉最后一个数量不足 batch size 的 batch.
# shuffle and batch dataset and drop the last batch shorter than batch_size
ds_train = ds_train.shuffle(1000).map(preprocess).batch(batch_size, drop_remainder=True)
ds_test = ds_test.shuffle(1000).map(preprocess).batch(batch_size, drop_remainder=True)


# output: <BatchDataset shapes: ((128, 80), (128, 46)), types: (tf.int32, tf.float32)>


# 使用动态路由(Dynamic Routing)聚合信息

# 1 将data(x)转换为vendor

# 2 使用rnn计算
class LSTM(tf.keras.Model):
    def __init__(self, num_units):
        super().__init__()
        # embedding [b, 200] -> [b, 200, 100]
        '''
          Input shape:(batch_size, max_news_words)
          Output shape:(batch_size, max_news_words, embedding_len)
          input_shape = (max_news_words,)
        '''
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_news_words)

        self.RNN1 = tf.keras.layers.LSTM(num_units, dropout=0.5, return_sequences=True)
        self.RNN2 = tf.keras.layers.LSTM(num_units, dropout=0.5, return_sequences=False)

        # self.fc = tf.keras.layers.Dense(32, activation=tf.tanh)

    def call(self, inputs, training=None):
        outputs = self.embedding(inputs)

        outputs = self.RNN1(outputs, training=training)
        outputs = self.RNN2(outputs, training=training)
        # -----------
        # outputs = self.fc(outputs)
        # print(outputs)
        # # output: Tensor("dense/Tanh:0", shape=(None, 32), dtype=float32)
        # maxlen = max_news_words  # ???
        # outputs = tf.reshape(outputs, shape=[batch_size, maxlen, out_caps_num, outputs.get_shape()[-1]])
        # -----------
        print(outputs)
        # Tensor("lstm_2/StatefulPartitionedCall:0", shape=(None, 100), dtype=float32)
        return outputs


model = LSTM(100)
model.build(input_shape=(None, max_news_words))
model.summary()
'''
output:
Model: "lstm"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        multiple                  8000000   
_________________________________________________________________
lstm_1 (LSTM)                multiple                  120400    
_________________________________________________________________
lstm_2 (LSTM)                multiple                  80400     
=================================================================
Total params: 8,200,800
Trainable params: 8,200,800
Non-trainable params: 0
_________________________________________________________________
'''
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 3 rnn输出 -全连接层-> 动态路由中的uhat
def shared_routing_uhat(caps,  # 输入 shape(b_sz, maxlen, caps_dim)
                        out_caps_num,  # 输出胶囊个数
                        scope=None):
    batchsize, maxlen = tf.shape(caps)[0], tf.shape(caps)[1]  # 获取批次和长度
    out_caps_dim = caps.get_shape()[-1]  # 输出胶囊维度
    caps_uhat = tf.keras.layers.Dense(caps, out_caps_num * out_caps_dim, activation=tf.tanh)
    caps_uhat = tf.reshape(caps_uhat, shape=[batchsize, maxlen, out_caps_num, out_caps_dim])
    return caps_uhat


u_hat = shared_routing_uhat(model.outputs, 5)


# 4 进行动态计算 得到最终的输出
def squashing(in_caps, axes):  # 定义squashing激活函数
    _EPSILON = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(in_caps), axis=axes, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + _EPSILON)
    vec_squashed = scalar_factor * in_caps  # element-wise
    return vec_squashed


def masked_routing_iter(caps_uhat, seqLen, iter_num):  # 动态路由计算
    batch_size, maxlen = tf.shape(caps_uhat)[0], tf.shape(caps_uhat)[1]  # 获取批次和长度
    out_caps_num = int(caps_uhat.get_shape()[2])
    seqLen = tf.where(tf.equal(seqLen, 0), tf.ones_like(seqLen), seqLen)

    B = tf.zeros([batch_size, maxlen, out_caps_num], dtype=tf.float32)
    for i in range(iter_num):
        C = tf.nn.softmax(B, axis=2)
        weighted_uhat = C * caps_uhat
        S = tf.reduce_sum(weighted_uhat, axis=1)

        A = squashing(S, axes=[2])
        A = tf.expand_dims(A, axis=1)  # 增加一维
        B = tf.reduce_sum(caps_uhat * A, axis=-1) + B  # 更新B
    return 0


def routing_masked():  # 胶囊网络
    masked_routing_iter(u_hat)
    return 0
