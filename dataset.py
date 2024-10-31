import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_dataloader(data_path, sequence_length, batch_size):
    # 加载数据集
    data = pd.read_csv(data_path)

    # 预处理数据
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    # 计算移动平均线 (MA)
    ma_period = sequence_length  # MA 的周期，设为与输入序列长度相同
    data['MA_20'] = data['Close'].rolling(window=ma_period).mean()

    # 计算指数平滑异同平均线 (MACD)
    short_ema_period = sequence_length  # 短期 EMA 的周期，设为与输入序列长度相同
    long_ema_period = sequence_length * 2  # 长期 EMA 的周期，设为输入序列长度的两倍
    signal_period = sequence_length // 2  # 信号线的周期，设为输入序列长度的一半

    data['EMA_12'] = data['Close'].ewm(span=short_ema_period, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_ema_period, adjust=False).mean()
    data['DIF'] = data['EMA_12'] - data['EMA_26']
    data['DEA'] = data['DIF'].ewm(span=signal_period, adjust=False).mean()
    data['MACD'] = 2 * (data['DIF'] - data['DEA'])

    # 计算相对强弱指数 (RSI)
    rsi_period = sequence_length  # RSI 的周期，设为与输入序列长度相同
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # 计算随机指标 (KDJ)
    kdj_period = sequence_length  # KDJ 的周期，设为与输入序列长度相同

    low_min = data['Low'].rolling(window=kdj_period).min()
    high_max = data['High'].rolling(window=kdj_period).max()

    data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['%D'] = data['%K'].rolling(window=3).mean()  # D 线是 K 线的 3 天移动平均
    data['%J'] = 3 * data['%K'] - 2 * data['%D']  # J 线根据 K 和 D 计算得出

    # 对价格和技术指标进行标准化
    scaler_standard = StandardScaler()
    features_price_tech = data[['Open', 'High', 'Low', 'Close', 'MA_20', 'MACD', 'RSI', '%K', '%D', '%J']]
    features_price_tech_scaled = scaler_standard.fit_transform(features_price_tech)

    # 对成交量和成交额进行归一化
    scaler_minmax = MinMaxScaler(feature_range=(0, 1))
    features_volume = data[['Volume', 'Turnover']]
    features_volume_scaled = scaler_minmax.fit_transform(features_volume)

    # 合并处理后的特征
    features_array = np.hstack((features_price_tech_scaled, features_volume_scaled))
    features_array = features_array[~np.isnan(features_array).any(axis=1)]  # 去除包含 NaN 的行

    # 将数据分为训练集和测试集（80% 训练集，20% 测试集）
    train_data, test_data = train_test_split(features_array, test_size=0.2, shuffle=False)

    # 定义自定义数据集类
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, sequence_length=10):  # sequence_length 表示输入序列的长度，设为 10
            self.data = data
            self.sequence_length = sequence_length

        def __len__(self):
            return len(self.data) - self.sequence_length

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.sequence_length]
            y = self.data[idx + self.sequence_length, 3]  # 使用第十一天的 'Close' 作为目标
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    train_dataset = TimeSeriesDataset(train_data, sequence_length=sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length=sequence_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练集数据加载器，打乱顺序
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集数据加载器，保持顺序

    return train_dataloader, test_dataloader, scaler_standard, scaler_minmax


# # 示例用法
# train_dataloader, test_dataloader, scaler_standard, scaler_minmax = create_dataloader('./data/SP500_10_years_data.csv')
#
# # 显示一个批次以验证训练数据加载器
# for x_batch, y_batch in train_dataloader:
#     print("输入批次形状:", x_batch.shape)
#     print("目标批次形状:", y_batch.shape)
#     break
