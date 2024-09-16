Repository for CCCWG: Cross-scale Coupled Correlative Wavelet Graphs for Joint Prediction of Multiple Financial Time Series

Raw data:
- ^GSPC.csv: S&P 500 time series
- ^N225.csv: N225 time series
- jpysdr.csv: JPY/SDR time series

Data preprocessing:
- process_data.py: prepocess the raw data
- process_mra.m: Matlab code for MODWT

Processed data:
- sp500_n225: standardized (S&P 500 and N225) time series with 5-level Haar-MODWT sequences
- sp500_oil: standardized (S&P 500 and WTI Oil) time series with 5-level Haar-MODWT sequences
- n225_jpy: standardized (N225 and JPY/SDR) time series with 5-level Haar-MODWT sequences

Model Implementation:
- crosscorr.py: code for Cross-scale Correlation Unit
- corrwavlet.py: code for the main model (Cross-scale Correlative Wavelet Networks)

Evaluation:
- eval_corrwavlet: the main entry to run the evaluation
- loadData: set variable "data_path" to load different training and testing datasets.
- metrics: code for evaluation metrics
