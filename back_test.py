import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lstm import build_lstm_model
from dataset import create_dataloader


class ModelBacktest:
    def __init__(self, model_path, data_path, threshold=37, initial_cash=100000, commission=0.001):
        """
        初始化回测参数。
        :param model_path: 训练好的模型文件路径
        :param data_path: 数据文件路径
        :param threshold: 买入决策阈值
        :param initial_cash: 初始现金
        :param commission: 交易佣金
        """
        self.model = self.load_model(model_path)
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        self.scaler = StandardScaler().fit(self.data[['Close']])
        self.threshold = threshold
        self.cash = initial_cash
        self.commission = commission
        self.positions = pd.Series([0] * len(self.data), index=self.data.index)
        self.portfolio_value = pd.Series([initial_cash] * len(self.data), index=self.data.index)

    def load_model(self, model_path):
        """
        加载 LSTM 模型。
        """
        model = build_lstm_model(12, 512, 2, 1, dropout=0.3)  # 确保模型结构与训练时一致
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 评估模式
        return model

    def predict(self, X):
        """
        使用模型进行预测。
        :param X: 输入数据
        :return: 预测值
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # 增加 batch 维度
        with torch.no_grad():
            output = self.model(X_tensor)
            return output[:, -1, :].item()  # 返回最后一个时间步的预测值

    def run_backtest(self):
        """
        执行基于 test_dataloader 的回测。
        """
        train_dataloader, test_dataloader, scaler_standard, scaler_minmax = create_dataloader('./data/SP500_10_years_data.csv', 10, 64)
        self.model.eval()  # 设置模型为评估模式
        for i, (X, y) in enumerate(test_dataloader):  # 按批次遍历回测数据
            # 模型预测
            with torch.no_grad():
                predictions = self.model(X)[:, -1, :].cpu().numpy().flatten()

            for j, prediction in enumerate(predictions):
                current_date = self.data.index[i * test_dataloader.batch_size + j]
                yesterday_close = self.data.loc[current_date, 'Close']

                # 买入条件：预测值 > 阈值
                if prediction > self.threshold:
                    shares = (self.cash * (1 - self.commission)) / yesterday_close
                    self.positions[current_date] += shares
                    self.cash -= shares * yesterday_close

                    # 卖出条件：次日收盘价卖出
                    next_date = self.data.index[i * test_dataloader.batch_size + j + 1]
                    next_close = self.data.loc[next_date, 'Close']
                    self.cash += shares * next_close * (1 - self.commission)
                    self.positions[next_date] -= shares

                # 更新组合价值
                current_position = self.positions[current_date] * yesterday_close
                self.portfolio_value[current_date] = self.cash + current_position

    def calculate_metrics(self):
        """
        计算回测指标。
        :return: 回测绩效指标，包括总收益率、最大回撤、年化收益率
        """
        returns = self.portfolio_value.pct_change().fillna(0)
        cumulative_return = self.portfolio_value[-1] / self.portfolio_value[0] - 1
        max_drawdown = (self.portfolio_value.cummax() - self.portfolio_value).max() / self.portfolio_value.cummax().max()
        annualized_return = (1 + cumulative_return) ** (252 / len(self.data)) - 1

        metrics = {
            'Cumulative Return': cumulative_return,
            'Max Drawdown': max_drawdown,
            'Annualized Return': annualized_return
        }
        return metrics


if __name__ == "__main__":
    # 初始化回测
    model_path = './model/lstm_model/MAE_36.6491_epoch_80.pth'
    data_path = './data/SP500_10_years_data.csv'
    backtest = ModelBacktest(model_path, data_path)

    # 运行回测
    backtest.run_backtest()

    # 输出回测结果
    metrics = backtest.calculate_metrics()
    print(metrics)
