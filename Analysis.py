
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

print('A min/max: ',min(df['A']),max(df['A']),"Mean: ", df['A'].mean(),"Std: ", df['A'].std())
print('A (c = 1) min/max: ',min(df2['A']),max(df2['A']), df2['A'].mean(), df2['A'].std())
print('A (c = -1 min/max: ',min(df3['A']),max(df3['A']), df3['A'].mean(), df3['A'].std())

main(data_path = 'data_569')

#function to calculate mutual information


