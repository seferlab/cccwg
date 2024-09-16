from corrwavlet import CorrelateWavelet
from sklearn.metrics import mean_squared_error, accuracy_score, auc
from loadData import *
import numpy as np
from metrics import *
import matplotlib.pyplot as pyplot
import scipy.io as spiocd

tgt = 0

wavs_a = matdata['wavs_a']
wavs_b = matdata['wavs_b']
wavs_a_t, wavs_a_y = sliding_windows(wavs_a, sq_len)
wavs_b_t, wavs_b_y = sliding_windows(wavs_b, sq_len)

if tgt == 0:
    train_wav_x = [wavs_a_t[:tr_data.shape[1] - sq_len], wavs_b_t[:tr_data.shape[1] - sq_len]]
    train_wav_y = wavs_a_y[:, :tr_data.shape[1] - sq_len]
    test_wav_x = [wavs_a_t[tr_data.shape[1] - sq_len:], wavs_b_t[tr_data.shape[1] - sq_len:]]
    test_wav_y = wavs_a_y[:, tr_data.shape[1] - sq_len:]
else:
    train_wav_x = [wavs_b_t[:tr_data.shape[1] - sq_len], wavs_a_t[:tr_data.shape[1] - sq_len]]
    train_wav_y = wavs_b_y[:, :tr_data.shape[1] - sq_len]
    test_wav_x = [wavs_b_t[tr_data.shape[1] - sq_len:], wavs_b_t[tr_data.shape[1] - sq_len:]]
    test_wav_y = wavs_b_y[:, tr_data.shape[1] - sq_len:]

y = [updown_train[tgt], train_y[tgt]] + np.split(train_wav_y.T, train_wav_y.shape[0], axis=1)

cowavelet = CorrelateWavelet()
## The parameters need to be tuned for different datasets
cowavelet.build(wave_size=(wavs_a.shape[0], sq_len),
                wave_scales=[64],
                wave_filters=32,
                wave_kernel_size=[4,5,6,7,8,8],
                using_timediff=False,
                conv_activation=None,
                using_autocross_feat=True,
                using_cross_feat=True,
                cross_feats=[25],
                direction_units=200,
                cross_activation='tanh',
                direction_activation='tanh',
                crosscorr_scale=1)
cowavelet.compile()

## Need to check the results for early stop
for it in range(5):
    cowavelet.value_model.fit(x=train_wav_x, y=y, batch_size=64, epochs=10)
    # cowavelet.value_model.fit(x=train_wav_x, y=y, batch_size=32, epochs=1, shuffle=False)

    result = cowavelet.value_model.predict(test_wav_x)
    c0 = np.squeeze(result[0])
    pred_wav = c0 > 0.5
    acc_wav = accuracy_score(updown_label[tgt], pred_wav)
    print("Acc: %.4f" % (acc_wav))

    c1 = np.squeeze(result[1]) * data_stds[tgt] + data_means[tgt]
    rmse_cowav = np.sqrt(mean_squared_error(test_y[tgt], c1))
    pred_cowav = np.squeeze((c1 - test_y_1[tgt, :-1]) > 0)
    acc_cowav = accuracy_score(updown_label[tgt], pred_cowav)
    mape_cowav = mean_absolute_percentage_error(test_y[tgt], c1)
    print("VAcc: %.4f, RMSE: %.4f, MAPE: %.4f" % (acc_cowav, rmse_cowav, mape_cowav))


    rr_cowav = return_rate(updown_rate[tgt], pred_wav)
    cum_rr_cowav= cum_return_rate(updown_rate[tgt], pred_wav)
    pyplot.figure(1)
    pyplot.plot(cum_rr_cowav)
    print("ARR: %.4f" % rr_cowav)

    profit_cowav= profit(updown[tgt], pred_wav)
    cum_profit_cowav= cum_profit(updown[tgt], pred_wav)
    pyplot.figure(2)
    pyplot.plot(cum_profit_cowav)

    pyplot.figure(1)
    rr_pefect = return_rate(updown_rate[tgt], updown_label[tgt])
    cum_rr_perfect = cum_return_rate(updown_rate[tgt], updown_label[tgt])
    pyplot.plot(cum_rr_perfect)

    pyplot.figure(2)
    profit_perfect = profit(updown[tgt], updown_label[tgt])
    cum_profit_perfect = cum_profit(updown[tgt], updown_label[tgt])
    pyplot.plot(cum_profit_perfect)

    pyplot.show()

    pyplot.plot(test_y[tgt])
    pyplot.plot(c1)
    pyplot.show()

spio.savemat('cowav.mat', {'cum_rr_cowav_jpy': cum_profit_cowav, 'perfect_rr_jpy': cum_profit_perfect})

