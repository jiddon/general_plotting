#!/usr/bin/python3
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np
import fire

def read_csv(filename):
    """
    Read nicely formatted csv into pandas dataframe.
    """
    df = pd.read_csv(filename, delimiter=',', skipinitialspace=True)
    return df

def normalise(df, exclude=None):
    result = df.copy()
    for col in df.columns:
        if col != exclude:
            max_value = df[col].max()
            min_value = df[col].min()
            if (max_value != 0 and min_value != 0):
                result[col] = (df[col] - min_value) / (max_value - min_value)
            else:
                result[col] = df[col]
    return result

class Plots(object):
    """
    Plotting methods.
    """
    def __init__(self, filename):
        self.filename = filename
        self._df = read_csv(filename)

    def print_headers(self):
        """
        Print csv file headers.
        """
        print(self._df)
        print("Available headers to plot: ")
        for header in self._df.columns.values:
            header.replace("'", "")
            print(f"    {header}")

    def hist(self, x, bins):
        """
        Histogram.
        """
        print(self._df)
        self._df.hist(column=x, bins=bins)
        plt.show()
        
    def scatter(self, x, y):
        """
        Plot scatter.
        """
        print(self._df)
        self._df.plot.scatter(x,y)
        plt.tight_layout()
        plt.show()

    def plot(self, x, y):
        """
        Line plot.
        """
        df = self._df.sort_values(by=x)
        print(df)
        df.plot(x,y)
        plt.tight_layout()
        plt.show()

    def plot_as_function_of(self, x):
        """
        Line plot for all columns as a function of x.
        """
        df = self._df.sort_values(by=x)
        print(df)
        df_cols = list(df.columns.values)
        df_cols.remove(x)
        ncols = len(df_cols)
        pcols = 3
        prows = ncols//pcols+1 if ncols%pcols>0 else ncols//pcols        
        fig, axes = plt.subplots(nrows=prows, ncols=pcols)
        axes[-((pcols*prows)-ncols), -((pcols*prows)-ncols)].axis('off')
        for n,col in enumerate(df_cols):
            df.plot(x, col, ax=axes[n//3, n%3])
        plt.tight_layout()
        plt.show()

    def plot_normalised_as_function_of(self, x):
        """
        Line plot for all normalised columns as a function of x.
        """
        _df = self._df.sort_values(by=x)
        df = normalise(_df, exclude=x)
        print(df)
        df_cols = list(df.columns.values)
        df_cols.remove(x)
        ncols = len(df_cols)
        ax = df.plot(x, df_cols[0])
        for n,col in enumerate(df_cols[1:]):
            df.plot(x, col, ax=ax)
        ax.set_ylabel("Normalised column", fontsize=10)
        plt.tight_layout()
        plt.show()

    def plot_same_canvas(self, x, *y):
        """
        Line plot for specified columns as a function of x.
        """
        df = self._df.sort_values(by=x)
        print(df)
        ax = df.plot(x, y[0])
        for n,col in enumerate(y[1:]):
            df.plot(x, col, ax=ax)
        ax.set_ylabel("Normalised column", fontsize=10)
        plt.tight_layout()
        plt.show()

    def correlation(self):
        """
        Correlation matrix.
        """
        corr = self._df.corr()
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        plt.figure()
        ax = plt.subplot()
        sns.heatmap(corr, cmap=cmap, square=True, annot=True)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.tight_layout()
        plt.show()

    def regression(self, x, y):
        """
        Linear regression.
        """
        df = self._df.sort_values(by=x)
        print(df)
        g = sns.jointplot(x=x, y=y, data=df, kind='reg')
        r, p = stats.pearsonr(df[x], df[y])
        g.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                           xy=(0.1, 0.9), xycoords='axes fraction',
                           ha='left', va='center',
                           bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
        plt.show()

    def pairplot(self, sort_on=None):
        """
        Pair plot.
        """
        if not sort_on:
            df = self._df
        else:
            df = self._df.sort_values(by=sort_on)            
        print(df)
        sns.set_theme(style="ticks")
        sns.pairplot(self._df)
        plt.show()

    def residuals(self, x, y):
        """
        Residuals plot.
        """
        print(self._df)
        sns.residplot(x=x, y=y, lowess=True, data=self._df)
        plt.show()

    def pie(self):
        """
        Pie chart.
        """
        print(self._df)
        labels = self._df.columns.values
        x = self._df.iloc[0,:]
        plt.figure()
        colours = sns.color_palette("pastel")[0:len(labels)]
        plt.pie(x, labels=labels, colors=colours, autopct="%1.1f%%")
        plt.tight_layout()
        plt.show()
        
if __name__=="__main__":
    fire.Fire(Plots)
    
