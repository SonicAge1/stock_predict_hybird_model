import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv('./data/SP500_10_years_data.csv')

history = list(data['Close'])
history = np.diff(history, n=1)

# 绘制残差的自相关函数（ACF）图
plt.figure(figsize=(10, 6))
plot_acf(history, lags=40)
plt.title('Residuals Autocorrelation (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# 计算 RACF（残差自相关函数）
plt.figure(figsize=(10, 6))
plot_pacf(history, lags=40, method='ywm')
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# 执行ADF（Augmented Dickey-Fuller）测试
result = adfuller(history)

# 提取ADF测试结果
adf_statistic = result[0]
p_value = result[1]
critical_values = result[4]

# 打印结果
print(f"ADF Statistic: {adf_statistic}")
print(f"p-value: {p_value}")
print("Critical Values:")
for key, value in critical_values.items():
    print(f"   {key}: {value}")
