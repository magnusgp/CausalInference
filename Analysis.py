
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(data = None):
    # read in the data



    df = pd.read_csv('sample/'+ data+'.csv', index_col = 0)
    #Pairplot
    sns.pairplot(df, height=1.5)
    plt.savefig('plots/Scatter_Pairplot_'+data+'.png')
    plt.show()
    sns.pairplot(df,kind ="kde", height=1.5)
    plt.savefig('plots/KDE_Pairplot.png_'+data+'.png')


    plt.show()
    print("Basic description of the data: ")
    print(df.describe())
    print()
    print("Correlation matrix: ")
    print(df.corr())




main(data = 'data_C=1')