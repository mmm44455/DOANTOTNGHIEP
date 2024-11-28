from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Concatenate, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping
from utils.attention_layers import DropAttention, GlobalSelfAttention
from tkinter import filedialog, messagebox, ttk
from utils.performance_metrics import show_model_performance
from keras.optimizers import Adam
def train_model(selected_model,X_train, y_train,X_test, y_test,performance_text):
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Dừng nếu không cải thiện sau 10 epochs
    restore_best_weights=True
)
    if selected_model.get() == "MultiAttention":
        # Mô hình MultiAttention
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        lstm_out = LSTM(64, return_sequences=True)(input_layer)
        attention = MultiHeadAttention(num_heads=15, key_dim=64)(lstm_out, lstm_out)
        dropout_layer = Dropout(0.1)(attention)
        lstm_out_attention = LSTM(32)(dropout_layer)
        output = Dense(1)(lstm_out_attention)
        model = Model(inputs=input_layer, outputs=output)
        
    elif selected_model.get() == "LSTM":
        model = Sequential([
                LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
                Dropout(0.2),
                LSTM(32, activation='relu', return_sequences=False),
                Dropout(0.2),
                Dense(1)  # Dự đoán mực nước (1 giá trị)
                ])
        
    elif selected_model.get() == "DropAttention":
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        lstm_out = LSTM(64, return_sequences=True)(input_layer)
        drop_attention_layer = DropAttention(d_model=64, num_heads=4, dropout_rate=0.1)
        attention_output, attention_weights = drop_attention_layer(lstm_out, lstm_out, lstm_out)
        concat_output = Concatenate()([lstm_out, attention_output])
        lstm_out_attention = LSTM(32)(concat_output)
        output_layer = Dense(1)(lstm_out_attention)
        model = Model(inputs=input_layer, outputs=output_layer)
        
    elif selected_model.get()=="GlobalSelfAttention":
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        lstm_out = LSTM(64, return_sequences=True)(input_layer)
        global_attention_layer = GlobalSelfAttention(d_model=64, num_heads=4)
        attention_output, attention_weights = global_attention_layer(lstm_out, lstm_out, lstm_out)
        concat_output = Concatenate()([lstm_out, attention_output])
        output_layer = Dense(1)(concat_output[:, -1, :])
        model = Model(inputs=input_layer, outputs=output_layer)
    
    if selected_model.get() in ["MultiAttention", "LSTM"]:
            model.compile(optimizer='adam', loss='mse')
    else:
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping],verbose=1,
    shuffle=True)
    messagebox.showinfo("Thông báo", "Huấn luyện mô hình thành công.")
    model.save('my_model.h5')
    show_model_performance(X_test,y_test,model,performance_text)
    