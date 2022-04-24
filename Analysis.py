
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def MI(x,y,Nbins = 21, plot = False):
    bins = np.linspace(min(x),max(x),Nbins)
    eps = np.spacing(1)
    x_marginal = np.histogram(x,bins)[0]
    y_marginal = np.histogram(y,bins)[0]
    x_marginal = x_marginal/x_marginal.sum()
    y_marginal = y_marginal/y_marginal.sum()
    xy_joint = np.histogram2d(x,y,bins = (bins,bins))[0]
    xy_joint = xy_joint/xy_joint.sum()
    MI = np.sum(xy_joint * np.log(xy_joint / (x_marginal[:, None] * y_marginal[None, :] + eps) + eps))
    if plot:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(xy_joint,origin = 'lower')
        plt.title('Joint Distribution')
        plt.subplot(1,2,2)
        plt.imshow((x_marginal[:,None]*y_marginal[None,:]).T,origin = 'lower')
        plt.title('Product of marginals')
        plt.suptitle('Mutual Information: '+str(MI))
        plt.show()
    return MI

def multiple_MI(df):

    cols = df.columns
    out = np.zeros((len(cols),len(cols)))
    for i in range(len(cols)):
        for j in range(len(cols)):
            out[i,j] = MI(df[cols[i]],df[cols[j]])
    return pd.DataFrame(out, columns=cols, index=cols)

def diffInStats(df1,df2, col):
    print()
    print("Basic description of the data before and after an intervention")#on {col}".format(col=col))"
    print("No intervention: ")
    print("mean: {mean:.3f}".format(mean = df1[col].mean())
          ,"std: {std:.3f}".format(std = df1[col].std())
          ,"min: {min:.3f}".format(min = df1[col].min())
          ,"max: {max:.3f}".format(max = df1[col].max()),
          "median: {median:.3f}".format(median = df1[col].median()))

    print("Intervention: ")
    print("mean: {mean:.3f}".format(mean = df2[col].mean())
              ,"std: {std:.3f}".format(std = df2[col].std())
              ,"min: {min:.3f}".format(min = df2[col].min())
              ,"max: {max:.3f}".format(max = df2[col].max()),
              "median: {median:.3f}".format(median = df2[col].median()))

    print("Difference: ")
    print("mean: {mean:.3f}".format(mean = abs(df2[col].mean() - df1[col].mean())),
          "std: {std:.3f}".format(std = abs(df2[col].std() - df1[col].std())),
          "min: {min:.3f}".format(min = abs(df2[col].min() - df1[col].min())),
          "max: {max:.3f}".format(max = abs(df2[col].max() - df1[col].max())),
          "median: {median:.3f}".format(median = abs(df2[col].median() - df1[col].median())))

def main(data_path,plot = False):

    df = pd.read_csv('sample/' + data_path + '.csv', index_col=0)

    if plot:
        #Pairplot
        sns.pairplot(df, height=1.5)
        plt.savefig('plots/Scatter_Pairplot_'+data_path+'.png')
        plt.show()
        sns.pairplot(df,kind ="kde", height=1.5)
        plt.savefig('plots/KDE_Pairplot.png_'+data_path+'.png')
        plt.show()


    print("Basic description of the data: ")
    print(df.describe())
    print()
    print("Correlation matrix: ")
    print(df.corr())
    print()
    print("Mutual Information Matrix: ")
    print(multiple_MI(df))




df = pd.read_csv('sample/data_569.csv', index_col = 0)
df2 = pd.read_csv('sample/data_C=1-57.csv', index_col = 0)
df3 = pd.read_csv('sample/data_C=-1-57.csv', index_col = 0)

#main(data_path = 'data_569')

diffInStats(df,df2,'A')




#function to calculate mutual information


