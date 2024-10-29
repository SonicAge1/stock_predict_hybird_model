class Config:
    class Data:
        file_path = './data/SP500_10_years_data.csv'  # 股票数据文件路径
        columns = ['Date', 'Close']  # 使用的数据列
        residuals_file = 'residuals.csv'  # 保存残差的文件路径

    class ARIMA:
        order = (1, 1, 1)  # ARIMA 模型的参数 (p, d, q)

    class MLP:
        input_size = 1
        hidden_size = 64
        output_size = 16

    class Transformer:
        seq_length = 20  # 序列长度
        batch_size = 8  # 批次大小
        feature_size = 16  # 特征维度大小
        num_heads = 4  # Transformer 多头注意力的头数
        num_layers = 2  # Transformer 编码器的层数
        hidden_dim = 64  # Transformer 编码器的隐藏层维度
        dropout = 0.1  # Transformer 模型的 dropout 概率

    class LSTM:
        input_size = 64  # LSTM 输入的特征维度
        hidden_size = 128  # LSTM 隐藏层维度
        num_layers = 2  # LSTM 层数
        output_size = 1  # LSTM 输出维度
        dropout = 0.1  # LSTM 模型的 dropout 概率

    class Training:
        epochs = 50  # 训练的轮次
        learning_rate = 0.001  # 学习率
