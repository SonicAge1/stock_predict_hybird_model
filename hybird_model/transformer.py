import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from config import Config

# 设置设备为CPU
device = torch.device('cpu')


# 规范残差数据
def create_data(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]  # 使用下一个时间步作为目标值

        # 将序列特征组合起来
        sequences.append(seq)
        labels.append(label)

    return torch.tensor(sequences), torch.tensor(labels, dtype=torch.float32).to(device)


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, feature_size, num_heads, num_layers, hidden_dim, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(feature_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 20, hidden_dim))  # 假设序列长度为 20
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.fc_out = nn.Linear(hidden_dim)

    def forward(self, x):
        # 添加嵌入层和位置编码
        # print(x.shape)
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        # Transformer 编码器部分
        x = self.transformer_encoder(x)
        # 修改 Transformer 输出为 LSTM 输入的形状
        # print(x.shape)
        return x


def load_data(file_path, seq_length, batch_size, test_split=0.2):
    # 加载残差数据
    residuals_df = pd.read_csv(file_path, index_col=0)
    residuals = residuals_df['0'].values

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    residuals_normalized = scaler.fit_transform(residuals.reshape(-1, 1))

    # 划分训练集和测试集
    split_index = int(len(residuals_normalized) * (1 - test_split))
    train_data = residuals_normalized[:split_index]
    test_data = residuals_normalized[split_index:]

    # 使用改进后的函数生成序列数据 - 训练集
    X_train, y_train = create_data(train_data, seq_length)
    # 使用改进后的函数生成序列数据 - 测试集
    X_test, y_test = create_data(test_data, seq_length)

    # 创建训练集和测试集的数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, scaler


def build_transformer_model(feature_size, num_heads, num_layers, hidden_dim, dropout):
    # 实例化 Transformer 模型
    transformer_model = TransformerModel(feature_size, num_heads, num_layers, hidden_dim, dropout).to(device)
    return transformer_model


# 打印模型和数据维度的辅助函数
def print_model_info(transformer_model, X, y):
    # 打印 Transformer 模型的结构
    print(transformer_model)
    # 打印数据维度
    print(X.shape, y.shape)