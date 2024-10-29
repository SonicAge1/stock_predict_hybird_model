import pandas as pd
import torch
from ARIMA import load_stock_data, arima_forecast, calculate_errors, save_residuals
from transformer import build_transformer_model, load_data as load_transformer_data
from lstm import build_lstm_model
from config import Config
import matplotlib.pyplot as plt
from MLP import MLP
import numpy as np


# 定义训练混合模型的函数
def train_mixed_model(mlp_model, transformer_model, lstm_model, train_dataloader, test_dataloader, scaler, config):
    criterion = torch.nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(list(transformer_model.parameters()) + list(lstm_model.parameters()), lr=config.Training.learning_rate)  # 优化器

    transformer_model.train()  # 设置 Transformer 模型为训练模式
    lstm_model.train()  # 设置 LSTM 模型为训练模式

    train_losses = []
    test_losses = []

    for epoch in range(config.Training.epochs):
        epoch_train_losses = []

        # 训练阶段
        for batch_idx, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()  # 梯度清零
            X = X.float()
            mlp_output = mlp_model(X)

            transformer_output = transformer_model(mlp_output)  # Transformer 前向传播

            lstm_output = lstm_model(transformer_output)  # LSTM 前向传播
            loss = criterion(lstm_output, y)  # 计算损失
            loss.backward(retain_graph=True)  # 反向传播
            optimizer.step()  # 更新参数

            epoch_train_losses.append(loss.item())

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        # 测试阶段
        if (epoch % 5) == 0:
            mlp_model.eval()  # 设置 MLP 模型为评估模式
            transformer_model.eval()  # 设置 Transformer 模型为评估模式
            lstm_model.eval()  # 设置 LSTM 模型为评估模式

            epoch_test_losses = []

            with torch.no_grad():
                for batch_idx, (X, y) in enumerate(test_dataloader):
                    X = X.float()
                    mlp_output = mlp_model(X)

                    transformer_output = transformer_model(mlp_output)  # Transformer 前向传播

                    lstm_output = lstm_model(transformer_output)  # LSTM 前向传播
                    loss = criterion(lstm_output, y)  # 计算损失

                    epoch_test_losses.append(loss.item())

            avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
            test_losses.append(avg_test_loss)

            print(f'Epoch [{epoch + 1}/{config.Training.epochs}], Train Loss: {avg_train_loss*100:.4f}, Test Loss: {avg_test_loss*100:.4f}')

            # 恢复模型为训练模式
            mlp_model.train()
            transformer_model.train()
            lstm_model.train()
        else:
            print(f'Epoch [{epoch + 1}/{config.Training.epochs}], Train Loss: {avg_train_loss*100:.4f}')

    # plt.plot(range(1, config.Training.epochs + 1), train_losses, label='Train Loss', color='blue')
    # plt.plot(range(1, len(test_losses) * 5 + 1, 5), test_losses, label='Test Loss', color='red')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # 计算反标准化后的测试集 MAE
    mlp_model.eval()
    transformer_model.eval()
    lstm_model.eval()

    all_mae = []
    all_y_unscaled = []
    all_lstm_output_unscaled = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_dataloader):
            X = X.float()
            mlp_output = mlp_model(X)

            transformer_output = transformer_model(mlp_output)  # Transformer 前向传播

            lstm_output = lstm_model(transformer_output)  # LSTM 前向传播

            # 反标准化
            lstm_output_unscaled = scaler.inverse_transform(lstm_output.cpu().numpy())
            y_unscaled = scaler.inverse_transform(y.cpu().numpy())

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
    train_dataloader, test_dataloader, scaler = load_transformer_data(
        file_path=config.Data.residuals_file,
        seq_length=config.Transformer.seq_length,
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
    train_mixed_model(mlp_model, transformer_model, lstm_model, train_dataloader, test_dataloader, scaler, config)


if __name__ == "__main__":
    main()
