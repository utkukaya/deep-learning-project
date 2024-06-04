from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    normalize_data = pd.DataFrame(scaled_data, columns=data.columns)
    return normalize_data
