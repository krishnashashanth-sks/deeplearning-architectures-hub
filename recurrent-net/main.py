from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utils import *

num_data_points=1000
raw_data=generate_time_series(num_data_points)
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(raw_data)

look_back=50
X,y=create_sequences(scaled_data,look_back)
X=np.reshape(X,(X.shape[0],X.shape[1],1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

model=Sequential()
model.add(Bidirectional(LSTM(units=100,return_sequences=True),input_shape=(look_back,1)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=100,return_sequences=False)))
model.add(Dropout(0.3))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
history=model.fit(
    X_train,y_train,
    epochs=50,
    batch_size=32,validation_split=0.2,
    verbose=1
)

# Make predictions on the test set
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE (Root Mean Squared Error)
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')

# Plot the results
plt.figure(figsize=(15, 7))

# Shift train predictions for plotting
train_plot = np.empty_like(raw_data)
train_plot[:, :] = np.nan
train_plot[look_back:len(train_predict) + look_back, :] = train_predict

# Shift test predictions for plotting
test_plot = np.empty_like(raw_data)
test_plot[:, :] = np.nan
# Corrected indexing for test_plot
test_plot[len(train_predict) + look_back : len(raw_data), :] = test_predict

# Plot original data, training predictions, and test predictions
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_plot, label='Training Prediction')
plt.plot(test_plot, label='Test Prediction')
plt.title('Time Series Forecasting with Stacked Bidirectional LSTM')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()