import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, MultiHeadAttention, Dropout, Concatenate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.dates as mdates 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.layers import Layer
# Hàm tải dữ liệu huấn luyện
sequence_length  = 1
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length, 0])  
    return np.array(sequences), np.array(targets)
def load_train_data():
    global X,y ,X_train, X_test, y_train, y_test, scaler
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        data.set_index('Ngay', inplace=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        # Chuẩn bị dữ liệu
        sequence_length = 1
        # X, y = [], []
        # for i in range(sequence_length, len(data_scaled)):
        #     X.append(data_scaled[i-sequence_length:i])
        #     y.append(data_scaled[i])

        # X, y = np.array(X), np.array(y)
        
        X, y = create_sequences(data_scaled, sequence_length)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        messagebox.showinfo("Thông báo", "Dữ liệu huấn luyện đã được tải thành công.")
class DropAttention(Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(DropAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)
        self.dropout = Dropout(self.dropout_rate)

    def split_heads(self, x, batch_size):
        """Chia chiều cuối cùng thành (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Tính toán attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Drop some attention weights
        attention_weights = self.dropout(attention_weights, training=True)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth)

        # Scaled dot-product attention with dropout applied
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len, d_model)

        return output, attention_weights
class GlobalSelfAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(GlobalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply softmax to get the attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len, d_model)

        return output, attention_weights

# Hàm tạo và huấn luyện mô hình
def train_model():
    global model, selected_model
    if selected_model.get() == "MultiAttention":
        # Mô hình MultiAttention
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        lstm_out = LSTM(64, return_sequences=True)(input_layer)
        attention = MultiHeadAttention(num_heads=15, key_dim=64)(lstm_out, lstm_out)
        dropout_layer = Dropout(0.1)(attention)
        attention_output = tf.reduce_sum(dropout_layer, axis=1)
        # lstm_out_attention = LSTM(32)(dropout_layer)
        output = Dense(1)(attention_output)
        model = Model(inputs=input_layer, outputs=output)
        
    elif selected_model.get() == "LSTM":
        model = Sequential()
        # Lớp đầu vào
        model.add(LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
        # Lớp đầu ra
        model.add(Dense(1))
        
    elif selected_model.get() == "DropAttention":
        input_layer = Input(shape=(X.shape[1], X.shape[2]))
        lstm_out = LSTM(64, return_sequences=True)(input_layer)
        drop_attention_layer = DropAttention(d_model=64, num_heads=4, dropout_rate=0.1)
        attention_output, attention_weights = drop_attention_layer(lstm_out, lstm_out, lstm_out)
        concat_output = Concatenate()([lstm_out, attention_output])
        lstm_out_attention = LSTM(32)(concat_output)
        output_layer = Dense(7)(lstm_out_attention)
        model = Model(inputs=input_layer, outputs=output_layer)
        
    elif selected_model.get()=="GlobalSelfAttention":
        input_layer = Input(shape=(X.shape[1], X.shape[2]))
        lstm_out = LSTM(64, return_sequences=True)(input_layer)
        global_attention_layer = GlobalSelfAttention(d_model=64, num_heads=4)
        attention_output, attention_weights = global_attention_layer(lstm_out, lstm_out, lstm_out)
        concat_output = Concatenate()([lstm_out, attention_output])
        output_layer = Dense(7)(concat_output[:, -1, :])
        model = Model(inputs=input_layer, outputs=output_layer)
    
    if selected_model.get() in ["MultiAttention", "LSTM"]:
            model.compile(optimizer='adam', loss='mse')
    else:
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    messagebox.showinfo("Thông báo", "Huấn luyện mô hình thành công.")
    show_model_performance()
    
# Hàm tải dữ liệu dự đoán
def load_predict_data():
    global X_new, Y_new, new_data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        new_data = pd.read_csv(file_path)
        new_data.set_index('Ngay', inplace=True)
        new_data_scaled = scaler.transform(new_data)

        sequence_length = 1
        X_new, Y_new = [], []
        for i in range(sequence_length, len(new_data_scaled)):
            X_new.append(new_data_scaled[i-sequence_length:i])
            Y_new.append(new_data_scaled[i])
        X_new = np.array(X_new)
        Y_new = np.array(Y_new)
        messagebox.showinfo("Thông báo", "Dữ liệu dự đoán đã được tải thành công.")
        
# Hàm dự đoán và hiển thị kết quả
def predict_and_show():
    predictions = model.predict(X_new)
    predictions_original = scaler.inverse_transform(predictions)[:, 0]
    y_test_original = scaler.inverse_transform(Y_new)[:, 0]
    # Hiển thị kết quả dự báo trong bảng
    # Cắt mảng ngày tháng từ sequence_length
    dates = new_data.index[sequence_length:]
    # Đảm bảo độ dài của mảng ngày, thực tế và dự báo khớp
    min_length = min(len(dates), len(predictions_original), len(y_test_original))
    dates = dates[:min_length]
    predictions_new_original = predictions_original[:min_length]
    y_test_new_original = y_test_original[:min_length]
    df_results = pd.DataFrame({
        'Ngày': dates,
        'Thực tế': y_test_new_original,
        'Dự đoán': predictions_new_original
    })
    for item in tree.get_children():
        tree.delete(item)

    # Thêm dữ liệu vào bảng
    for row in df_results.itertuples():
        tree.insert("", tk.END, values=row[1:])
def showChart():
    predictions = model.predict(X_new)
    predictions_original = scaler.inverse_transform(predictions)[:, 0]
    y_test_original = scaler.inverse_transform(Y_new)[:, 0]
    min_length = min(len(new_data.index), len(y_test_original), len(predictions_original))
    dates = new_data.index[:min_length]
    dates = pd.to_datetime(dates, format='%d/%m/%Y')
    predictions_new_original = predictions_original[:min_length]
    y_test_new_original = y_test_original[:min_length]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_test_new_original, label='Thực tế', color='blue')
    ax.plot(dates, predictions_new_original, label='Dự báo', color='red')
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Mực nước')

    # Định dạng ngày tháng
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend()
    fig.autofmt_xdate()
    canvas = FigureCanvasTkAgg(fig, master=root)  # root là cửa sổ Tkinter
    canvas.draw()
    canvas.get_tk_widget().pack()
    
# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Dự đoán Mực nước")

# Tiêu đề
title_label = tk.Label(root, text="Ứng dụng Dự đoán Mực nước", font=("Arial", 16))
title_label.pack(pady=10)

# Nút tải dữ liệu huấn luyện
load_train_button = tk.Button(root, text="Tải dữ liệu huấn luyện", command=load_train_data)
load_train_button.pack(pady=5)

# Phần chọn mô hình huấn luyện
selected_model = tk.StringVar(value="MultiAttention")
model_options = ["MultiAttention", "DropAttention", "GlobalSelfAttention", "LSTM"]
model_menu = ttk.Combobox(root, textvariable=selected_model, values=model_options)
model_menu.pack(pady=5)

# Nút huấn luyện mô hình
train_button = tk.Button(root, text="Huấn luyện mô hình", command=train_model)
train_button.pack(pady=5)

# Hiển thị chất lượng mô hình
performance_text = tk.StringVar()
performance_label = tk.Label(root, textvariable=performance_text, font=("Arial", 12))
performance_label.pack(pady=5)

# Nút tải dữ liệu dự đoán
load_predict_button = tk.Button(root, text="Tải dữ liệu dự đoán", command=load_predict_data)
load_predict_button.pack(pady=5)

tree = ttk.Treeview(root, columns=list(["Ngày", "Thực tế", "Dự đoán"]), show="headings")

# Định nghĩa các cột
for col in ["Ngày", "Thực tế", "Dự đoán"]:
    tree.heading(col, text=col)
    tree.column(col, width=150, anchor='center')

# Đặt Treeview trong cửa sổ
tree.pack(expand=True, fill='both')
# Nút dự đoán
predict_button = tk.Button(root, text="Bảng dự đoán", command=predict_and_show)
predict_button.pack(pady=10)



predict_button = tk.Button(root, text="Biểu đồ dự đoán ", command=showChart)
predict_button.pack(pady=10)

# Chạy giao diện Tkinter
root.mainloop()