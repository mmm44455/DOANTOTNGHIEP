import numpy as np

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length, 0])  # Chỉ lấy giá trị cột đầu tiên
    return np.array(sequences), np.array(targets)
