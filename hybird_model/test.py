import pandas as pd
from sklearn.preprocessing import MinMaxScaler

residuals_df = pd.read_csv('./residuals.csv', index_col=0)
residuals = residuals_df['0'].values

scaler = MinMaxScaler(feature_range=(0, 1))
residuals_normalized = scaler.fit_transform(residuals.reshape(-1, 1))

print(residuals_normalized.shape)

# 划分训练集和测试集
split_index = int(len(residuals_normalized) * (1 - 0.2))

train_data = residuals[:split_index][0]
test_data = residuals[split_index:][0]

train_data = residuals_normalized[:split_index]
test_data = residuals_normalized[split_index:]

# train_data = scaler.inverse_transform(train_data)
# test_data = scaler.inverse_transform(test_data)

sequences = []
labels = []
for i in range(len(test_data) - 5):
    seq = test_data[i:i + 5]
    label = test_data[i + 5]  # 使用下一个时间步作为目标值

    # 将序列特征组合起来
    sequences.append(seq)
    labels.append(label)

print(sequences)
print(labels)