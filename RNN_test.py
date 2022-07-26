# 具有LSTM的RNN具有检测简单回归模型不能找到的关系和模式的能力。因此，如果RNN LSTM能够在崩盘之前学习到复杂价格结构，那么这种模型难道不应该胜过先前的测试模型吗？
#
#
#
# 为了回答这个问题，我们用了Python库中的Keras的LSTM实现了两个不同的RNN并经过了严格的超参数调优。首先确定的是每个层的输入序列的长度。每个时间步骤t的输入序列由之前连续几天到t天的每天的价格变化组成。由于较长的输入序列需要更多的内存并会减慢计算速度，因此必须谨慎地选择这个序列的长度。理论上，RNN LSTM应该能够找到长时间序列（前后两者之间的）依赖关系，然而，在Keras中的LSTM实现中，如果参数状态被设置为真，则单元状态仅从一个序列传递到下一个序列。在实践中，这种实现是繁琐的。为了避免在训练中，针对不同期的不同数据集，神经网络识别出长项依赖性，我就在训练数据切换数据集时手动重置状态。这个算法没有给出很强的结果，所以我把状态设置为false，将序列长度从5步增加到10步，并且从时间窗口向网络中输入平均价格变化和平均波动的额外序列，时间窗口从10个交易日直到252个交易日。（类似于先前测试模型所选择的特征）。最后，我对超参数进行调参，并尝试了不同的损失函数、层数、每层的神经元数和是否dropout（译者注：dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。）。
#
#
#
# # 性能最好的RNN LSTM具有顺序层，后跟两个LSTM层，每个层具有50个神经元，最后一层使用adam优化器（译者注：Adam 是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重。）、二元交叉熵损失函数和sigmoid激活函数。
# model_name = 'RNN LSTM'
# neurons = 50
# dropout = 0
# optimizer = 'adam'
# loss = 'binary_crossentropy'
# activation = 'sigmoid'
# stateful = True
# inp_dim = 1   # <-- 1 if price change only, 2 if volatility as well
# inp_tsteps = sequence + 4 * additional_feat
# def rnn_lstm(inp_tsteps, inp_dim, neurons, dropout):
#     model = Sequential()
#     model.add(LSTM(neurons, batch_input_shape=(batch_size, inp_tsteps, inp_dim), \
#             stateful=stateful, return_sequences=True))
#     model.add(LSTM(neurons, stateful=stateful, return_sequences=False))
#     model.add(Dense(3, activation=activation))
#     return model
# model = rnn_lstm(neurons=neurons, inp_tsteps=inp_tsteps, inp_dim=inp_dim, dropout=dropout)
# model.compile(loss=loss, optimizer=optimizer)
# model.summary()