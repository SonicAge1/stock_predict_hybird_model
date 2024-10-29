import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def load_stock_data(file_path, columns):
    # 读取数据集
    data = pd.read_csv(file_path)
    df = data[columns]
    # 数据预处理 - 填充缺失值
    df.fillna(method='ffill', inplace=True)
    # 将索引转换为日期格式
    df.index = pd.to_datetime(data['Date'], format='%Y-%m-%d')

    # 自动划分 7:3 训练集和测试集
    split_index = int(len(df) * 0.7)
    training_set = df.iloc[:split_index, :]
    test_set = df.iloc[split_index:, :]

    return training_set, test_set


def plot_train_test(training_set, test_set):
    # 绘制训练集和测试集
    plt.figure(figsize=(10, 6))
    plt.plot(training_set['Close'], label='Train Set')
    plt.plot(test_set['Close'], label='Test Set')
    plt.title('Stock Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()


def arima_forecast(training_set, test_set, order=(1, 1, 1)):
    history = list(training_set['Close'])
    predictions = []

    for t in range(len(test_set)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=1)[0]
        predictions.append(yhat)
        obs = test_set['Close'].iloc[t]
        history.append(obs)
        print(f'epoch({t}/{len(test_set)})')

    return predictions


def calculate_errors(test_set, predictions):
    mse = mean_squared_error(test_set['Close'], predictions)
    mae = mean_absolute_error(test_set['Close'], predictions)
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    return mse, mae


def save_residuals(test_set, predictions, file_path):
    residuals = pd.Series([test_set['Close'].iloc[i] - predictions[i] for i in range(len(predictions))], index=test_set.index)
    # 保存残差到 CSV 文件
    residuals.to_csv(file_path)
    # 绘制残差的自相关函数（ACF）图
    plt.figure(figsize=(10, 6))
    plot_acf(residuals, lags=40)
    plt.title('Residuals Autocorrelation (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()
    return residuals



