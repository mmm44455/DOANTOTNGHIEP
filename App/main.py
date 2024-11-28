import tkinter as tk
from tkinter import ttk
from utils.load_data import load_train_data, load_predict_data
from utils.model import train_model
from utils.attention_layers import DropAttention,GlobalSelfAttention
from utils.update_predict import update_data, predict_and_show, showChart
from tensorflow.keras.models import load_model
# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Dự đoán Mực nước")

# Tiêu đề
title_label = tk.Label(root, text="Ứng dụng Dự đoán Mực nước", font=("Arial", 16))
title_label.pack(pady=10)
# Biến lưu trữ dữ liệu
train_data = {"X_train": None, "X_test": None, "y_train": None, "y_test": None,"X":None,"y":None}
predict_data = {"X_test_new": None, "y_test_new": None,"new_data":None,"data_scaled":None,"model":None}

# Nút tải dữ liệu huấn luyện
def handle_load_train_data():
    X_train, X_test, y_train, y_test,X,y = load_train_data()
    if X_train is not None:
        train_data["X_train"], train_data["X_test"] = X_train, X_test
        train_data["y_train"], train_data["y_test"] = y_train, y_test
        
# Nút tải dữ liệu huấn luyện
load_train_button = tk.Button(root, text="Tải dữ liệu huấn luyện", command=handle_load_train_data)
load_train_button.pack(pady=5)

# Phần chọn mô hình huấn luyện

selected_model = tk.StringVar(value="MultiAttention")
model_options = ["MultiAttention", "DropAttention", "GlobalSelfAttention", "LSTM"]
model_menu = ttk.Combobox(root, textvariable=selected_model, values=model_options)
model_menu.pack(pady=5)

# Nút huấn luyện mô hình
def handle_train_model():
    if train_data["X_train"] is None or train_data["y_train"] is None:
        tk.messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu huấn luyện trước!")
        return
    train_model(selected_model, train_data["X_train"], train_data["y_train"],train_data["X_test"],train_data["y_test"],performance_text)
# Nút huấn luyện mô hình
train_button = tk.Button(root, text="Huấn luyện mô hình", command=handle_train_model)
train_button.pack(pady=5)

# Hiển thị chất lượng mô hình

performance_text = tk.StringVar()
performance_label = tk.Label(root, textvariable=performance_text, font=("Arial", 12))
performance_label.pack(pady=5)

def handle_load_predict_data():
    X_test_new, y_test_new, new_data, data_scaled = load_predict_data()
    model = load_model('my_model.h5')
    if X_test_new is not None:
        # Lưu dữ liệu và mô hình vào predict_data để có thể sử dụng ở các bước sau
        predict_data["X_test_new"] = X_test_new
        predict_data["y_test_new"] = y_test_new
        predict_data["new_data"] = new_data
        predict_data["data_scaled"] = data_scaled
        predict_data["model"] = model
# Nút tải dữ liệu dự đoán
load_predict_button = tk.Button(root, text="Tải dữ liệu dự đoán", command=handle_load_predict_data)
load_predict_button.pack(pady=5)

# Bảng hiển thị kết quả dự đoán
tree = ttk.Treeview(root, columns=["Ngày", "Thực tế", "Dự đoán"], show="headings")
for col in ["Ngày", "Thực tế", "Dự đoán"]:
    tree.heading(col, text=col)
    tree.column(col, width=150, anchor='center')
tree.pack(expand=True, fill='both')

predict_button = tk.Button(root, text="Bảng dự đoán", command=lambda: [update_data(tree,predict_data), predict_and_show(tree,predict_data["new_data"])])
predict_button.pack(pady=10)

# Nút hiển thị biểu đồ
chart_button = tk.Button(root, text="Biểu đồ dự đoán", command=lambda: [update_data(tree,predict_data), showChart(root,predict_data["new_data"])])
chart_button.pack(pady=10)

# Chạy giao diện Tkinter
root.mainloop()
