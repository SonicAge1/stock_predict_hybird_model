import torch
import torch.nn as nn
from dataset import create_dataloader
from config import Config as config
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np

# 设置设备为CPU
device = torch.device('cpu')


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.fc_in = nn.Linear(input_size, 64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 部分
        x = self.fc_in(x)
        x, _ = self.lstm(x)
        # 输出层（取序列最后一个时间步）
        x = self.fc_out(x)
        return x


def build_lstm_model(input_size, hidden_size, num_layers, output_size, dropout=0.1):
    # 实例化 LSTM 模型
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    return lstm_model


def train():
    model = build_lstm_model(12, 512, 3, 1, dropout=0.3)
    train_dataloader, test_dataloader, scaler_standard, scaler_minmax = create_dataloader('./data/SP500_10_years_data.csv', 10, 64)
    train_losses = []
    test_losses = []
    criterion = torch.nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Training.learning_rate)  # 优化器

    for epoch in range(config.Training.epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        epoch_train_losses = []
        lstm_outputs_all = []
        y_all = []

        # 训练阶段
        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)  # 将数据移动到 GPU
            optimizer.zero_grad()  # 梯度清零
            X = X.float()

            # LSTM 前向传播
            lstm_output = model(X)
            # 只保留最后一个时间步的输出
            lstm_output = lstm_output[:, -1, :]
            y = y.view(-1, 1)
            loss = criterion(lstm_output, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            epoch_train_losses.append(loss.item())

            # 保存所有预测值和实际值
            lstm_outputs_all.extend(lstm_output.cpu().detach().numpy().flatten())
            y_all.extend(y.cpu().detach().numpy().flatten())

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        if (epoch % 1) == 0:
            model.eval()  # 设置模型为评估模式
            epoch_test_losses = []
            test_lstm_outputs_all = []
            test_y_all = []

            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(test_dataloader):
                    X, y = X.to(device), y.to(device)
                    X = X.float()
                    lstm_output = model(X)
                    lstm_output = lstm_output[:, -1, :]
                    y = y.view(-1, 1)
                    loss = criterion(lstm_output, y)
                    epoch_test_losses.append(loss.item())

                    # 保存测试集的所有预测值和实际值
                    test_lstm_outputs_all.extend(lstm_output.cpu().detach().numpy().flatten())
                    test_y_all.extend(y.cpu().detach().numpy().flatten())

            avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
            test_losses.append(avg_test_loss)
            model.train()

            # 计算 R^2 值
            r2 = r2_score(test_y_all, test_lstm_outputs_all)

            # 将预测值和实际值扩展到与原始特征形状一致
            test_lstm_outputs_all_extended = np.array(test_lstm_outputs_all).reshape(-1, 1)
            test_y_all_extended = np.array(test_y_all).reshape(-1, 1)

            # 重复扩展列，以匹配 scaler 的输入维度
            test_lstm_outputs_all_extended = np.repeat(test_lstm_outputs_all_extended, scaler_standard.n_features_in_, axis=1)
            test_y_all_extended = np.repeat(test_y_all_extended, scaler_standard.n_features_in_, axis=1)

            # 去标准化并提取第一列的结果
            test_lstm_outputs_all_inverse = scaler_standard.inverse_transform(test_lstm_outputs_all_extended)[:, 0]
            test_y_all_inverse = scaler_standard.inverse_transform(test_y_all_extended)[:, 0]

            # 计算 MAE
            mae = mean_absolute_error(test_y_all_inverse, test_lstm_outputs_all_inverse)

            print(f'Epoch [{epoch + 1}/{config.Training.epochs}], Train Loss: {avg_train_loss * 100:.4f}, Test Loss: {avg_test_loss * 100:.4f}, R^2: {r2:.4f}, MAE:{mae}')

            # 保存模型
            model_path = f'./model/lstm_model/MAE_{mae:.4f}_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)

            # 测试集预测值和实际值的散点图
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(test_y_all)), test_y_all, color='blue', alpha=0.6, label='Actual Values')
            plt.scatter(range(len(test_lstm_outputs_all)), test_lstm_outputs_all, color='orange', alpha=0.6, label='Predicted Values')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Test Predicted vs Actual Values Scatter Plot (Epoch {})'.format(epoch + 1))
            plt.savefig('./data/figure/test_predicted_vs_actual_scatter_epoch_{}.png'.format(epoch + 1))
            plt.close()

train()