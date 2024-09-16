import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler


standardize = True
means = [0, 0]
stds = [1, 1]

df_sp500 = pd.read_csv('^GSPC.csv')
df_oil = pd.read_csv('WTIOil.csv')

df_join = df_sp500.merge(df_oil, on='Date')
test_idx = (df_join['Date'] >= '2019-01-01') & (df_join['Date'] <= '2019-12-31')
train_idx = (df_join['Date'] >= '2000-01-01') & (df_join['Date'] <= '2018-12-31')
train_set = df_join[train_idx]
test_set = df_join[test_idx]

train_data = np.vstack([train_set['Adj Close'], train_set['Price']])
test_data = np.vstack([test_set['Adj Close'], test_set['Price']])

all_data = np.hstack([train_data, test_data])

if standardize:
    scaler = StandardScaler()
    std_data = scaler.fit_transform(all_data.T)
    [train_data, test_data] = np.hsplit(std_data.T, [train_data.shape[1]])
    means = scaler.mean_
    stds = scaler.scale_

sio.savemat('sp500_oil.mat', {'train_data':train_data, 'test_data':test_data,
                              'means':means, 'stds':stds})


df_n225 = pd.read_csv('^N225.csv')
df_n225.dropna(subset=['Adj Close'], inplace=True)
df_n225['Adj Close'].astype(np.float64)
df_join = df_sp500.merge(df_n225, on='Date')

test_idx = (df_join['Date'] >= '2019-01-01') & (df_join['Date'] <= '2019-12-31')
train_idx = (df_join['Date'] >= '2000-01-01') & (df_join['Date'] <= '2018-12-31')
train_set = df_join[train_idx]
test_set = df_join[test_idx]

train_data = np.vstack([train_set['Adj Close_x'], train_set['Adj Close_y']])
test_data = np.vstack([test_set['Adj Close_x'], test_set['Adj Close_y']])

all_data = np.hstack([train_data, test_data])

if standardize:
    scaler = StandardScaler()
    std_data = scaler.fit_transform(all_data.T)
    [train_data, test_data] = np.hsplit(std_data.T, [train_data.shape[1]])
    means = scaler.mean_
    stds = scaler.scale_

sio.savemat('sp500_n225.mat', {'train_data':train_data, 'test_data':test_data,
                              'means':means, 'stds':stds})



df_jpy= pd.read_csv('jpysdr.csv')
df_jpy.dropna(subset=['Price'], inplace=True)
df_join = df_n225.merge(df_jpy, on='Date')

test_idx = (df_join['Date'] >= '2019-01-01') & (df_join['Date'] <= '2019-12-31')
train_idx = (df_join['Date'] >= '2000-01-01') & (df_join['Date'] <= '2018-12-31')
train_set = df_join[train_idx]
test_set = df_join[test_idx]

train_data = np.vstack([train_set['Adj Close'], train_set['Price']])
test_data = np.vstack([test_set['Adj Close'], test_set['Price']])

all_data = np.hstack([train_data, test_data])

if standardize:
    scaler = StandardScaler()
    std_data = scaler.fit_transform(all_data.T)
    [train_data, test_data] = np.hsplit(std_data.T, [train_data.shape[1]])
    means = scaler.mean_
    stds = scaler.scale_

sio.savemat('n225_jpy.mat', {'train_data':train_data, 'test_data':test_data,
                              'means':means, 'stds':stds})

