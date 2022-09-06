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

    def correlation(self, sort_on=None):
        """
        Correlation matrix.
        """
        if not sort_on:
            print(self._df)
        else:
            df = self._df.sort_values(by=sort_on)
            print(df)
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

    def pairplot(self):
        """
        Pair plot.
        """
        print(self._df)
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
        


if __name__=="__main__":
    fire.Fire(Plots)
    
