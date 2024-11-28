import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from utils.load_data import inverse_transform_data
scaler = MinMaxScaler(feature_range=(0, 1))
sequence_length = 10
def update_data(tree,predict_data):
    global predictions_new_original, y_test_new_original
    X_test_new = predict_data["X_test_new"]
    model = predict_data["model"]
    y_test_new = predict_data["y_test_new"]
    data_scaled = predict_data["data_scaled"]

    predictions_new = model.predict(X_test_new)
    predictions_new_reshaped = predictions_new.reshape(-1, predictions_new.shape[-1])  # Chuyển thành 2D
    if not hasattr(scaler, 'scale_'):  # Kiểm tra xem scaler đã được fit chưa
        scaler.fit(data_scaled)  # Fit scaler với data_scaled nếu chưa fit
    # Số cột từ scaler
    num_columns = data_scaled.shape[1]

    # Tạo mảng đủ cột để kết hợp
    zeros_to_add = np.zeros((predictions_new_reshaped.shape[0], num_columns - predictions_new_reshaped.shape[1]))

    # Chuyển đổi giá trị dự đoán về kích thước ban đầu
    predictions_new_original = inverse_transform_data(
        np.concatenate((predictions_new_reshaped, zeros_to_add), axis=1)
    )[:, 0]

    y_test_new_original = y_test_new.reshape(-1, 1)  # Chuyển thành 2D nếu cần
    zeros_to_add_y_test = np.zeros((y_test_new_original.shape[0], num_columns - 1))

    # Chuyển đổi giá trị thực tế về kích thước ban đầu
    y_test_new_original = inverse_transform_data(
        np.concatenate((y_test_new_original, zeros_to_add_y_test), axis=1)
    )[:, 0]

# Hàm dự đoán và hiển thị kết quả
def predict_and_show(tree,new_data):
    # Hiển thị kết quả dự báo trong bảng
    # Cắt mảng ngày tháng từ sequence_length
    dates = new_data.index[sequence_length:]
    # Đảm bảo độ dài của mảng ngày, thực tế và dự báo khớp
    min_length = min(len(dates), len(predictions_new_original), len(y_test_new_original))
    dates = dates[:min_length]
    predictions_to_show = predictions_new_original[:min_length]
    y_to_show = y_test_new_original[:min_length]
    df_results = pd.DataFrame({
        'Ngày': dates,
        'Thực tế': y_to_show,
        'Dự đoán': predictions_to_show
    })
    for item in tree.get_children():
        tree.delete(item)

    # Thêm dữ liệu vào bảng
    for row in df_results.itertuples():
        tree.insert("", tk.END, values=row[1:])
canvas = None
def showChart(root,new_data):
    global canvas
    min_length = min(len(new_data.index), len(y_test_new_original), len(predictions_new_original))
    dates = new_data.index[:min_length]
    dates = pd.to_datetime(dates, format='%d/%m/%Y')
    predictions_original = predictions_new_original[:min_length]
    y_test_original =y_test_new_original[:min_length]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_test_original, label='Thực tế', color='blue')
    ax.plot(dates, predictions_original, label='Dự báo', color='red')
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Mực nước')

    # Định dạng ngày tháng
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend()
    fig.autofmt_xdate()
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=root)  # root là cửa sổ Tkinter
    canvas.draw()
    canvas.get_tk_widget().pack()