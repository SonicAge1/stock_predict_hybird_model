import torch
import torch.nn as nn


# 设置设备为CPU
device = torch.device('cpu')


# 定义 LSTM 模型，用于预测残差值
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 部分
        x, _ = self.lstm(x)
        # 输出层（取序列最后一个时间步）
        x = self.fc_out(x[:, -1, :])
        return x


def build_lstm_model(input_size=32, hidden_size=32, num_layers=1, output_size=1, dropout=0.1):
    # 实例化 LSTM 模型
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    return lstm_model


# 打印模型和数据维度的辅助函数
def print_lstm_model_info(lstm_model, X, y):
    # 打印 LSTM 模型的结构
    print(lstm_model)
    # 打印数据维度
    print(X.shape, y.shape)
