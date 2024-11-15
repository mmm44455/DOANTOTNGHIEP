import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
import tensorflow_addons as tfa  # thư viện cho Attention
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense,Input, MultiHeadAttention, Dropout, Concatenate,LayerNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
data = pd.read_csv(r'C:\Users\57\Desktop\DOANTOTNGHIEP\Data\KichBan1_2\train_data2.csv') 
data.set_index('Ngay', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length, 0])  
    return np.array(sequences), np.array(targets)
sequence_length = 1 # Sử dụng số ngày trước để dự báo ngày tiếp theo
X, y = create_sequences(data_scaled, sequence_length)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
model = Sequential()

# Lớp đầu vào
# model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=False))
# # Lớp đầu ra
# model.add(Dense(1))

input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm_out = LSTM(64, activation='relu', return_sequences=True)(input_layer)
attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
# Lớp Attention
dropout_layer = Dropout(0.2)(attention_out)
lstm_out_attention = LSTM(32)(dropout_layer)
# Lớp đầu ra
output = Dense(1)(lstm_out_attention)
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
y_pred = model.predict(X_test)
# Chuyển đổi kích thước y_pred để phù hợp với y_test
y_pred = y_pred.flatten()  # Chuyển y_pred thành một mảng 1 chiều

y_test = y_test.flatten() # Biến đổi thành mảng 2 chiều

r2 = r2_score(y_test, y_pred)
print("R² (R-squared):", r2)

#Tính MAE
mae = mean_absolute_error(y_test, y_pred)
print("MAE (Mean Absolute Error):", mae)

# Tính RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE (Root Mean Square Error):", rmse)

# Tính NSE
nse_value = nse(y_test, y_pred)
print("NSE (Nash-Sutcliffe Efficiency):", nse_value)