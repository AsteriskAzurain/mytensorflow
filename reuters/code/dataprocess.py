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
# 将样本数据按照指定批次制作成tf.data.Dataset接口的数据集 并将不足一批次的剩余数据丢弃
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

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# output:(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)