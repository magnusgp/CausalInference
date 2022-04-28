import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


data = pd.read_csv("sample/data_279.csv", index_col=0)

# print("Columns:")
# print(data.columns)
# print("\n")
# print(data.head())

# Function to plot histograms for each column
def plot_hist(data, cols):
    # Apply seaborn style
    plt.style.use("seaborn")
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    plt.subplot(2, 3, 1)
    data.loc[:, cols[0]].hist(ax=axes[0, 0])
    plt.legend([cols[0]])
    plt.subplot(2, 3, 2)
    data.loc[:, cols[1]].hist(ax=axes[0, 1])
    plt.legend([cols[1]])
    plt.subplot(2, 3, 3)
    data.loc[:, cols[2]].hist(ax=axes[0, 2])
    plt.legend([cols[2]])
    plt.subplot(2, 3, 4)
    data.loc[:, cols[3]].hist(ax=axes[1, 0])
    plt.legend([cols[3]])
    plt.subplot(2, 3, 5)
    data.loc[:, cols[4]].hist(ax=axes[1, 1])
    plt.legend([cols[4]])
    plt.subplot(2, 3, 6)
    data.loc[:, cols[5]].hist(ax=axes[1, 2])
    plt.legend([cols[5]])
    plt.show()

# plot correlation matrix
def plot_corr(data):
    # Apply seaborn style
    plt.style.use("seaborn")
    # Create a correlation matrix
    corr = data.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

# Mutual Information
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

# mutual information Multiple
def multiple_MI(df):
    cols = df.columns
    out = np.zeros((len(cols),len(cols)))
    for i in range(len(cols)):
        for j in range(len(cols)):
            out[i,j] = MI(df[cols[i]],df[cols[j]])
    return pd.DataFrame(out, columns=cols, index=cols)

# Function to plot scatter plots for each column
def plot_scatter(data, cols):
    # Apply seaborn style
    plt.style.use("seaborn")
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    plt.subplot(2, 3, 1)
    plt.plot(data["A"], '.')
    plt.legend([cols[0]])
    plt.subplot(2, 3, 2)
    plt.plot(data["B"], '.')
    plt.legend([cols[1]])
    plt.subplot(2, 3, 3)
    plt.plot(data["C"], '.')
    plt.legend([cols[2]])
    plt.subplot(2, 3, 4)
    plt.plot(data["D"], '.')
    plt.legend([cols[3]])
    plt.subplot(2, 3, 5)
    plt.plot(data["E"], '.')
    plt.legend([cols[4]])
    plt.subplot(2, 3, 6)
    plt.plot(data["F"], '.')
    plt.legend([cols[5]])
    plt.show()

# Plot pairwise scatter plots using sns
def plot_pairwise(data, cols, kind = "scatter"):
    # Apply seaborn style
    plt.style.use("seaborn")
    # Create a scatter plot of the pairwise relationships
    sns.pairplot(data, vars=cols, size=3, aspect=1.5, kind=kind)
    plt.show()

if __name__ == "__main__":
    plot_pairwise(data, data.columns, kind = "scatter")
    # plot_corr(data)
