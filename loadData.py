import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import numpy as np

data_dir = r'./'
data_path = data_dir + r'sp500_n225.mat'
# data_path = data_dir + r'sp500_oil.mat'
# data_path = data_dir + r'n225_jpy.mat'
sq_len = 64


def sliding_windows(data, seq_length, y_start=None):
    x = [data[:, i:(i + seq_length)] for i in range((data.shape[1])-seq_length)]
    if y_start is None:
        y_start = seq_length
    y = data[:, y_start:]
    return np.array(x), y


matdata = sio.loadmat(data_path)

data_means = matdata['means'].T
data_stds = matdata['stds'].T

tr_data = matdata['train_data']
ts_data = np.hstack([tr_data[:, -sq_len:], matdata['test_data']])

train_x, train_y_1 = sliding_windows(tr_data, sq_len, sq_len-1)
test_x, test_y_1 = sliding_windows(ts_data, sq_len, sq_len-1)

updown_train = (train_y_1[:, 1:] - train_y_1[:, :-1]) > 0
train_y = train_y_1[:, 1:]

test_y_1 = test_y_1 * data_stds + data_means

updown = (test_y_1[:, 1:] - test_y_1[:, :-1])
updown_rate = updown / test_y_1[:, :-1]
updown_label = updown > 0

test_y = test_y_1[:, 1:]




