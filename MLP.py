import torch
import torch.nn as nn
from config import Config


# 定义一个简单的多层感知机 (MLP) 模型，将输入的形状 [8, 5, 1] 转换为 [8, 5, 10]
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),  # 输入维度为1，隐藏层大小为16
            nn.ReLU(),  # 激活函数
            nn.Linear(16, output_dim)  # 输出维度为10
        )

    def forward(self, x):
        # x 的形状为 [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        # 仅提取最后一个维度的元素，形状变为 [batch_size * seq_len, input_dim]
        x = x.view(-1, input_dim)
        # 全连接层计算
        x = self.fc(x)
        # 调整形状为 [batch_size, seq_len, output_dim]
        x = x.view(batch_size, seq_len, -1)
        return x

