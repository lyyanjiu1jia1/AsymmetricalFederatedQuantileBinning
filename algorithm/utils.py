import pandas as pd


def load_data(col_index):
    file_path = r'../data/breast_a.csv'
    data = pd.read_csv(file_path).to_numpy()
    col = data[:, col_index + 1]
    return col
