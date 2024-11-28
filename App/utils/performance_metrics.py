import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def nse(y_true, y_pred):
    return 1 - sum((y_true - y_pred) ** 2) / sum((y_true - np.mean(y_true)) ** 2)

def show_model_performance(X_test,y_test,model,performance_text):
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nse_value = nse(y_test.flatten(), y_pred.flatten())

    performance_text.set(f"R²: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | NSE :{nse_value:.3f}")

