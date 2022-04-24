import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def current_guess(image):
    img = mpimg.imread(image)
    imgplot = plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    data = "./sample/data_435.csv"
    data_CSV = CSV_load(data)
    plot_col_hist(data_CSV, 4)
    current_guess("./plots/artwork.jpg")