本项目首先通过tf.keras.datasets.reuters接口加载数据集，<br>
该数据集包含有11228条新闻，共分成46个主题，

使用tf.keras.preprocessing.sequence.pad_sequences函数，<br>
对于长度不足80个词的句子，在后面补0；对于长度超过80个词的句子，从前面截断，只保留80个词。

做好数据预处理后，定义数据集，<br>
将样本数据按照指定批次制作成tf.data.Dataset接口的数据集，<br>
并将不足一批次的剩余数据丢弃。

然后用动态路由算法聚合信息，<br>
注意将胶囊网络中的动态路由算法应用在RNN模型中还需要一些改动：<br>
（1）定义函数shared_routing_uhat，使用全连接网络，将RNN模型输出结果转换成动态路由中的uhat。<br>
（2）定义函数masked_routing_iter,进行动态路由计算。在该函数的开始部分，对输入的序列长度进行掩码处理，使动态路由算法支持动态长度的序列数据输入。<br>
（3）定义函数routing_masked完成全部动态路由计算过程，对RNN模型的输出结果进行信息聚合。然后对动态路由计算结果进行dropout处理，使其具有更强的泛化能力。

然后通过用IndyLSTM单元搭建RNN模型：<br>
（1）将3层IndyLSTM单元传入tf.nn.dynamic_rnn函数中，搭建动态RNN模型。<br>
（2）用函数routing_masked对RNN模型的输出结果做基于动态路由的信息聚合。<br>
（3）将聚合后的结果输入全连接网络，进行分类处理。<br>
（4）用分类后的结果计算损失值，并定义优化器用于训练。

最后建立会话，训练网络：<br>
用tf.data数据集接口的Iterator.from_structure方法获取迭代器，<br>
并按照数据集的遍历次数训练模型。

---

1. 任务1 明确项目需求与目标
1. 任务2 环境准备：tensorflow
1. 任务3 准备样本
    1. 任务3.1数据读取与预处理—对齐列数据并计算长度（作业代码）
    1. 任务3.2定义数据集批次，丢弃剩余数据（作业代码）
1. 任务4 用动态路由算法聚合信息。
    1. 任务4.1 定义函数shared_routing_uhat，使用全连接网络，将RNN模型输出结果转换成动态路由中的uhat。（作业代码）
    1. 任务4.2定义函数masked_routing_iter，计算变长RNN模型的掩码。（作业代码）
    1. 任务4.3定义函数routing_masked,完成全部动态路由计算过程，对RNN模型的输出结果进行信息聚合。然后对动态路由计算结果进行dropout处理，使其具有更强的泛化能力。（作业代码）
1. 任务5 用IndyLSTM单元搭建RNN模型。
    1. 任务5.1将3层IndyLSTM单元传入tf.nn.dynamic_rnn函数中，搭建动态RNN模型。（作业代码）
    1. 任务5.2用函数routing_masked对RNN模型的输出结果做基于动态路由的信息聚合。（作业代码）
    1. 任务5.3将聚合后的结果输出全连接网络，进行分类处理。（作业代码）
    1. 任务5.4用分类后的结果计算损失值，并定义优化器用于训练。（作业代码）
1. 任务6 建立会话，训练网络并测试。（作业代码）
1. 任务7 完成项目报告（作业）

---

### RNN模型建立
- 在 keras 当中我们有两种方式建立 RNN 模型，比较推荐的方式是调用 layers.SimpleRNN 类，比较简单，不需要手动处理层与层之间的状态信息。
- 另一种方式是调用 layers.SimpleRNNCell，这种方式比较底层，需要手动处理层与层之间的状态信息。这种方式虽然麻烦，但是有利于加深对 RNN 的理解。

---

### 构建RNN网络
1. 设置循环神经网络的**超参数**<br>

   ```python
   vocabolary_size = 5000 # 词汇表达大小
   sequence_length = 150 # 序列长度
   embedding_size = 64 #词向量大小
   num_hidden_units = 256 # LSTM细胞隐藏层大小
   num_fc1_units = 64 # 第1个全连接下一层的大小
   dropout_keep_probabilitr = 0.5 # dropout保留比例
   num_classes = 10 # 类别数量
   learning_rate = 1e-3 # 学习率0. 001
   batch_size = 64 #每批训练大小
   ```

1. 将每个样本统一长度为seq_length，train_X = kr.preprocessing.sequence.pad_sequences(train_idlist_list, sequence_length)。

1. 调用LabelEncoder对象的fit_transform方法做标签编码，代码为：train_y = labelEncoder.fit_transform(train_label_list)，格式为： [0 0 0 ... 9 9 9]，将训练数据的类别标签转换成整型。

1. 调用keras.untils库的to_categorical方法将标签编码的结果再做Ont-Hot编码，将整型标签转换成onehot，代码为：train_Y = kr.utils.to_categorical(train_y, num_classes)，格式为：[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.] … [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]。

1. 调用tf库的get_variable方法实例化可以更新的模型参数embedding，矩阵形状为vocabulary_size*embedding_size，即5000*64。代码为embedding = tf.get_variable('embedding', [vocabolary_size, embedding_size])。

1. 使用tf.nn库的embedding_lookup方法将输入数据做词嵌入，代码为：embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)。X_holder中已经设置了序列长度为150/600。得到新变量embedding_inputs的形状为batch_size*sequence_length*embedding_size。

1. **RNN层的搭建**：将上述六步中处理好的数据输入到LSTM网络中，调用的是tf.nn.rnn_cell.BasicLSTMCell 函数，代码为：lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_units), num_hidden_units为隐藏层神经元的个数，实验中设置为了128/256。还要设置一个 dropout 参数，避免过拟合，代码为：lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)。

1. LSTM cell 和三维的数据输入到 tf.nn.dynamic_rnn ，目的为展开整个网络并构建一整个 RNN 模型，代码为：outputs, state = tf.nn.dynamic_rnn(lstm_cell, embedding_inputs,dtype=tf.float32)。

1. 获取最后一个细胞的h，即最后一个细胞的短时记忆矩阵，代码为：last_cell = outputs[:, -1, :]。

1. 添加第一个**全连接层**：调用tf.layers.dense方法，将结果赋值给变量full_connect1，形状为batch_size*num_fc1_units，词向量大小与全连接层神经元一致。

1. 调用tf.contrib.layers.dropout方法，防止过拟合，代码为：full_connect1_dropout = tf.contrib.layers.dropout<br>
(full_connect1, dropout_keep_probability)。

1. 调用tf.nn.relu方法，即激活函数,增强拟合复杂函数的能力，代码为：full_connect1_activate = tf.nn.relu(full_connect1_dropout)。

1. 添加第二个**全连接层**：操作类似于第一个全连接层，但全连接层的神经元个数为10（对应新闻的10种类别），然后使用Softmax函数，将结果转化成10个类别的概率。

1. 使用交叉熵作为损失函数，调用tf.train.AdamOptimizer方法定义优化器optimizer  学习率设置为了0.001。
    ```python
    #使用交叉熵作为损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder, logits=full_connect2)
    loss = tf.reduce.mean(cross_entropy)
    #调用tf.train.AdamOptimizer方法定义优化器optimizer学习率设置为了0.001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    ```
##### 所以该文本识别模型为：输入数据-->RNN层（LSTM模型）-->全连接层1-->全连接层2 -->输出

