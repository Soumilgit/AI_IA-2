import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import holidays

power_df = pd.read_csv('PJME_hourly.csv', index_col=[0], parse_dates=[0])

holiday_map = holidays.US(years=power_df.index.year.unique())
power_df['holiday_flag'] = power_df.index.map(lambda dt: 1 if dt in holiday_map else 0)

power_df['hour_of_day'] = power_df.index.hour
power_df['day_of_week'] = power_df.index.dayofweek
power_df['month_of_year'] = power_df.index.month
power_df['year_value'] = power_df.index.year

plt.figure(figsize=(15, 5))
plt.plot(power_df['PJME_MW'], label='PJM East')
plt.title('PJM East Hourly Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power Consumption (MW)')
plt.legend()
plt.show()

feature_array = power_df[['PJME_MW', 'holiday_flag', 'hour_of_day', 'day_of_week', 'month_of_year']].values
data_normalizer = MinMaxScaler(feature_range=(0, 1))
normalized_features = data_normalizer.fit_transform(feature_array)

split_index = int(len(normalized_features) * 0.8)
training_set = normalized_features[:split_index]
testing_set = normalized_features[split_index:]

def build_sequences(dataset, time_steps=1):
    sequence_X, sequence_y = [], []
    for j in range(len(dataset) - time_steps - 1):
        sequence_X.append(dataset[j:(j + time_steps), :])
        sequence_y.append(dataset[j + time_steps, 0])
    return np.array(sequence_X), np.array(sequence_y)

time_lag = 24
X_train_seq, y_train_seq = build_sequences(training_set, time_lag)
X_test_seq, y_test_seq = build_sequences(testing_set, time_lag)

X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2])
X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2])

forecast_model = Sequential()
forecast_model.add(LSTM(100, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
forecast_model.add(Dropout(0.2))
forecast_model.add(LSTM(50, return_sequences=False))
forecast_model.add(Dropout(0.2))
forecast_model.add(Dense(1))

forecast_model.compile(optimizer='adam', loss='mean_squared_error')
forecast_model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32)

train_forecast = forecast_model.predict(X_train_seq)
test_forecast = forecast_model.predict(X_test_seq)

train_forecast_rescaled = data_normalizer.inverse_transform(
    np.concatenate((train_forecast, np.zeros((train_forecast.shape[0], feature_array.shape[1] - 1))), axis=1)
)[:, :1]

test_forecast_rescaled = data_normalizer.inverse_transform(
    np.concatenate((test_forecast, np.zeros((test_forecast.shape[0], feature_array.shape[1] - 1))), axis=1)
)[:, :1]

train_rmse = np.sqrt(mean_squared_error(y_train_seq, train_forecast_rescaled))
test_rmse = np.sqrt(mean_squared_error(y_test_seq, test_forecast_rescaled))
print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

plt.figure(figsize=(15, 5))
plt.plot(power_df.index[:split_index], data_normalizer.inverse_transform(training_set[time_lag:]), label='Train Data')
plt.plot(power_df.index[split_index + time_lag:], data_normalizer.inverse_transform(testing_set[time_lag:]), label='Test Data')
plt.plot(power_df.index[split_index + time_lag:], test_forecast_rescaled, label='Predicted Data', color='red')
plt.title('Electricity Consumption Forecasting')
plt.xlabel('Date')
plt.ylabel('Power Consumption (MW)')
plt.legend()
plt.show()
