
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



df = pd.read_csv('sample/data_569.csv', index_col = 0)
df2 = pd.read_csv('sample/data_C=1-57.csv', index_col = 0)
df3 = pd.read_csv('sample/data_C=-1-57.csv', index_col = 0)

print('A min/max: ',min(df['A']),max(df['A']),"Mean: ", df['A'].mean(),"Std: ", df['A'].std())
print('A min/max: ',min(df2['A']),max(df2['A']), df2['A'].mean(), df2['A'].std())
print('A min/max: ',min(df3['A']),max(df3['A']), df3['A'].mean(), df3['A'].std())

#main(data = 'data_C=-1-57')