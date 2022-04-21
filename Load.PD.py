import csv
import pandas as pd
import matplotlib.pyplot as plt

def CSV_load(filename):
    df = pd.read_csv(filename)
    return df

def plot_all(df):
    df = df.iloc[:, 1:]
    df.plot()
    plt.show()

def plot_col_hist(df, col):
    df = df.iloc[:, col]
    df.hist()
    plt.show()




if __name__ == '__main__':
    data = "./sample/data_445.csv"
    data_CSV = CSV_load(data)
    plot_col_hist(data_CSV, 4)