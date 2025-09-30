import pandas as pd
import numpy as np

datafile = "joint_dataset/assetprices.csv"

def read_data(file):
    data = pd.read_csv(file)
    array= np.array(data.iloc[:, 2:].columns)
    return array.astype(float)


def daily_log_returns(data):
    return np.log(data[1:] / data[:-1]) * 100

data = read_data(datafile)
returns = daily_log_returns(data)
