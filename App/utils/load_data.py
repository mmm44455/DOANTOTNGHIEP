import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tkinter import filedialog, messagebox
from utils.sequence_utils import create_sequences


sequence_length = 10
scaler = MinMaxScaler(feature_range=(0, 1))
# Hàm tải dữ liệu huấn luyện
def load_train_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        data.set_index('Ngay', inplace=True)
        data_scaled = scaler.fit_transform(data)

        X, y = create_sequences(data_scaled, sequence_length)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        messagebox.showinfo("Thông báo", "Dữ liệu huấn luyện đã được tải thành công.")
        return X_train, X_test, y_train, y_test,X,y
# Hàm tải dữ liệu dự đoán
def load_predict_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        new_data = pd.read_csv(file_path)
        new_data.set_index('Ngay', inplace=True)
        data_scaled = scaler.transform(new_data)
        X_test_new, y_test_new = create_sequences(data_scaled, sequence_length)
        messagebox.showinfo("Thông báo", "Dữ liệu dự đoán đã được tải thành công.")
        return X_test_new, y_test_new, new_data, data_scaled
def inverse_transform_data(data_scaled):
    # Giải mã dữ liệu từ 0-1 về lại phạm vi ban đầu
    return scaler.inverse_transform(data_scaled)