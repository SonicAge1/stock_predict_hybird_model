import pandas as pd
import torch
from dataset import create_dataloader
from ARIMA import load_stock_data, arima_forecast, calculate_errors, save_residuals
from transformer import build_transformer_model, load_re_data as load_transformer_data
from lstm import build_lstm_model
from config import Config
import matplotlib.pyplot as plt
from MLP import MLP
import numpy as np


def train_mixed_model(mlp_model, transformer_model, lstm_model, train_dataloader, test_dataloader, scaler_standard, scaler_minmax, config):
    # 使用 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model = mlp_model.to(device)
    transformer_model = transformer_model.to(device)
    lstm_model = lstm_model.to(device)

    criterion = torch.nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(list(transformer_model.parameters()) + list(lstm_model.parameters()), lr=config.Training.learning_rate)  # 优化器

    transformer_model.train()  # 设置 Transformer 模型为训练模式
    lstm_model.train()  # 设置 LSTM 模型为训练模式

    train_losses = []
    test_losses = []

    for epoch in range(config.Training.epochs):
        epoch_train_losses = []
        lstm_outputs_all = []
        y_all = []

        # 训练阶段
        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)  # 将数据移动到 GPU

            optimizer.zero_grad()  # 梯度清零
            X = X.float()
            mlp_output = mlp_model(X)

            transformer_output = transformer_model(mlp_output)  # Transformer 前向传播
            lstm_output = lstm_model(transformer_output)  # LSTM 前向传播

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

        if (epoch % 5) == 0:
            # 绘制预测值和实际值的散点图
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(y_all)), y_all, color='blue', alpha=0.6, label='Actual Values')
            plt.scatter(range(len(lstm_outputs_all)), lstm_outputs_all, color='orange', alpha=0.6, label='Predicted Values')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Training Predicted vs Actual Values Scatter Plot (Epoch {})'.format(epoch + 1))
            plt.savefig('./data/figure/training_predicted_vs_actual_scatter_epoch_{}.png'.format(epoch + 1))
            plt.close()

        if (epoch % 5) == 0:
            mlp_model.eval()  # 设置 MLP 模型为评估模式
            transformer_model.eval()  # 设置 Transformer 模型为评估模式
            lstm_model.eval()  # 设置 LSTM 模型为评估模式

            epoch_test_losses = []

            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(test_dataloader):
                    X, y = X.to(device), y.to(device)  # 将数据移动到 GPU
                    X = X.float()
                    mlp_output = mlp_model(X)
                    transformer_output = transformer_model(mlp_output)  # Transformer 前向传播
                    lstm_output = lstm_model(transformer_output)  # LSTM 前向传播

                    y = y.view(-1, 1)
                    loss = criterion(lstm_output, y)  # 计算损失
                    epoch_test_losses.append(loss.item())

            # 计算平均测试损失
            avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
            test_losses.append(avg_test_loss)

            # 恢复模型为训练模式
            mlp_model.train()
            transformer_model.train()
            lstm_model.train()
            print(f'Epoch [{epoch + 1}/{config.Training.epochs}], Train Loss: {avg_train_loss * 100:.4f}, Test Loss: {avg_test_loss * 100:.4f}')

        else:
            print(f'Epoch [{epoch + 1}/{config.Training.epochs}], Train Loss: {avg_train_loss * 100:.4f}')

    # 计算反标准化后的测试集 MAE
    mlp_model.eval()
    transformer_model.eval()
    lstm_model.eval()

    all_mae = []
    all_y_unscaled = []
    all_lstm_output_unscaled = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)  # 将数据移动到 GPU
            y = y.view(-1, 1)
            X = X.float()

            mlp_output = mlp_model(X)
            transformer_output = transformer_model(mlp_output)  # Transformer 前向传播
            lstm_output = lstm_model(transformer_output)  # LSTM 前向传播

            close_mean = scaler_standard.mean_[3]  # 获取 'Close' 的均值
            close_scale = scaler_standard.scale_[3]  # 获取 'Close' 的标准差

            lstm_output_unscaled = lstm_output.cpu().numpy() * close_scale + close_mean
            y_unscaled = y.cpu().numpy() * close_scale + close_mean

            # 计算 MAE
            mae = np.mean(np.abs(lstm_output_unscaled - y_unscaled))
            all_mae.append(mae)

            all_y_unscaled.extend(y_unscaled)
            all_lstm_output_unscaled.extend(lstm_output_unscaled)

    avg_mae = sum(all_mae) / len(all_mae)
    print(f'Final MAE on unscaled test set: {avg_mae:.4f}')


def main():
    config = Config()

    # # 第一步：加载 ARIMA 模型所需的股票数据
    # training_set, test_set = load_stock_data(
    #     file_path=config.Data.file_path,
    #     columns=config.Data.columns
    # )
    #
    # # 第二步：训练 ARIMA 模型并保存残差
    # arima_predictions = arima_forecast(training_set, test_set, order=config.ARIMA.order)
    # calculate_errors(test_set, arima_predictions)
    # residuals = save_residuals(test_set, arima_predictions, config.Data.residuals_file)

    # 第三步：将 ARIMA 残差构建为MLP输入
    # train_dataloader, test_dataloader, scaler = load_transformer_data(
    #     file_path=config.Data.residuals_file,
    #     seq_length=config.Transformer.seq_length,
    #     batch_size=config.Transformer.batch_size
    # )

    train_dataloader, test_dataloader, scaler_standard, scaler_minmax = create_dataloader(
        data_path=config.Data.file_path,
        sequence_length=config.Transformer.seq_length,
        batch_size=config.Transformer.batch_size
    )

    # 第三步：构建 MLP 模型
    mlp_model = MLP(
        input_dim=config.MLP.input_size,
        output_dim=config.MLP.output_size
    )

    # 第四步：构建 Transformer 模型
    transformer_model = build_transformer_model(
        feature_size=config.Transformer.feature_size,
        num_heads=config.Transformer.num_heads,
        num_layers=config.Transformer.num_layers,
        hidden_dim=config.Transformer.hidden_dim,
        dropout=config.Transformer.dropout
    )

    # 第五步：构建 LSTM 模型
    lstm_model = build_lstm_model(
        input_size=config.LSTM.input_size,
        hidden_size=config.LSTM.hidden_size,
        num_layers=config.LSTM.num_layers,
        output_size=config.LSTM.output_size,
        dropout=config.LSTM.dropout
    )

    # 第九步：训练混合模型
    train_mixed_model(mlp_model, transformer_model, lstm_model, train_dataloader, test_dataloader, scaler_standard, scaler_minmax, config)


if __name__ == "__main__":
    main()
