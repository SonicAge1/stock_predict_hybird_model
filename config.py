class Config:
    class Data:
        file_path = './data/SP500_10_years_data.csv'  # 股票数据文件路径
        columns = ['Date', 'Close']  # 使用的数据列
        residuals_file = 'residuals.csv'  # 保存残差的文件路径

    class ARIMA:
        order = (1, 1, 1)  # ARIMA 模型的参数 (p, d, q)

    class MLP:
        input_size = 12
        hidden_size = 128
        output_size = 256

    class Transformer:
        seq_length = 20  # 序列长度：20。增加序列长度，帮助模型捕捉更长的时间依赖。
        batch_size = 64  # 批次大小：32。适中批次大小，以减少显存占用，且易于训练收敛。
        feature_size = 256  # 特征维度大小：64。更大的特征维度有助于提升表示能力。
        num_heads = 8  # Transformer 多头注意力的头数：8。更多的注意力头以捕捉不同的模式。
        num_layers = 3  # Transformer 编码器的层数：3。适中深度，兼顾建模能力和训练难度。
        hidden_dim = 512  # Transformer 编码器的隐藏层维度：128。较大的隐藏层维度帮助学习更复杂的特征。
        dropout = 0.3  # Transformer 模型的 dropout 概率：0.2。防止过拟合。

    class LSTM:
        input_size = 512  # LSTM 输入的特征维度：与 Transformer 输出保持一致。
        hidden_size = 256  # LSTM 隐藏层维度：256。较大的隐藏层维度有助于记忆长时间依赖。
        num_layers = 2  # LSTM 层数：2。适中层数，避免过深导致训练困难。
        output_size = 1  # LSTM 输出维度：1。单一预测值。
        dropout = 0.3  # LSTM 模型的 dropout 概率：0.2。防止过拟合。

    class Training:
        epochs = 200  # 训练的轮次
        learning_rate = 0.0001  # 学习率
