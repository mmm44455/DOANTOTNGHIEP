import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, MultiHeadAttention, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Hàm để tải và chuẩn hóa dữ liệu
def load_data(file_path, sequence_length=7):
    global X_train, X_test, y_train, y_test, scaler
    data = pd.read_csv(file_path, parse_dates=['Ngay'], dayfirst=True)
    data.set_index('Ngay', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i, 0])  # Giả sử bạn dự đoán 'mucNuocHN' (cột đầu tiên)
    
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Xây dựng mô hình LSTM + Attention
def create_model(input_shape):
    global model
    input_layer = Input(shape=input_shape)
    
    # Lớp LSTM đầu tiên
    lstm_out = LSTM(64, return_sequences=True)(input_layer)
    
    # Lớp Multi-Head Attention
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
    concat_output = Concatenate()([lstm_out, attention])
    dropout_layer = Dropout(0.1)(concat_output)
    
    # Lớp LSTM thứ hai
    lstm_out_attention = LSTM(32)(dropout_layer)
    
    # Lớp đầu ra
    output = Dense(1)(lstm_out_attention)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(), loss='mse')
    
    return model

# Huấn luyện mô hình
def train_model(file_path):
    X_train, X_test, y_train, y_test, scaler = load_data(file_path)
    
    model = create_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, scaler, history

# Hàm đánh giá mô hình
def evaluate_model(model, X_test, y_test, scaler):
    # Thực hiện dự đoán trên tập kiểm tra
    predictions = model.predict(X_test)
    
    # Chuyển ngược giá trị dự đoán và thực tế về thang đo ban đầu
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], X_test.shape[2] - 1))), axis=1))[:, 0]
    y_test_scaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2] - 1))), axis=1))[:, 0]
    
    # Tính toán các độ đo đánh giá
    mae = mean_absolute_error(y_test_scaled, predictions)
    mse = mean_squared_error(y_test_scaled, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_scaled, predictions)
    
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Percentage Error (MAPE):", mape)

# Hàm vẽ biểu đồ dự đoán
def plot_predictions(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], X_test.shape[2] - 1))), axis=1))[:, 0]
    y_test_scaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_test.shape[2] - 1))), axis=1))[:, 0]

    plt.plot(y_test_scaled, label='Thực tế')
    plt.plot(predictions, label='Dự đoán')
    plt.xlabel('Thời gian')
    plt.ylabel('Mực nước')
    plt.legend()
    plt.show()

# Đường dẫn tới tệp dữ liệu
file_path = 'C:\\Users\\57\\Desktop\\DOANTOTNGHIEP\\Data\\KichBan1_2\\test_data2_1.csv'

# Huấn luyện mô hình và đánh giá
model, scaler, history = train_model(file_path)
evaluate_model(model, X_test, y_test, scaler)
plot_predictions(model, X_test, y_test, scaler)
