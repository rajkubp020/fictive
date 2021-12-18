from .plot import *
import matplotlib

def set_mood_dark():
    params = {
        'figure.edgecolor': 'white',
        'figure.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.facecolor': 'black',
        'savefig.edgecolor': 'black',
        'savefig.facecolor': 'black',
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'boxplot.boxprops.color': 'white',
        'boxplot.capprops.color': 'white',
        'boxplot.flierprops.color': 'white',
        'boxplot.flierprops.markeredgecolor': 'white',
        'boxplot.whiskerprops.color': 'white',
        'grid.color': '#e0e0e0',
        'hatch.color': 'white',
        'image.cmap': 'hot',
        'patch.edgecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
    }
    matplotlib.pyplot.rcParams.update(params)

def set_mood_light():
    linestyles()
    c1 = 'black'
    c2 = 'white'
    params = {
        'figure.edgecolor': c1,
        'figure.facecolor': c2,
        'axes.edgecolor': c1,
        'axes.facecolor': c2,
        'savefig.edgecolor': c1,
        'savefig.facecolor': c2,
        'text.color': c1,
        'axes.labelcolor': c1,
        'boxplot.boxprops.color': c1,
        'boxplot.capprops.color': c1,
        'boxplot.flierprops.color': c1,
        'boxplot.flierprops.markeredgecolor': c1,
        'boxplot.whiskerprops.color': c1,
        'grid.color': '#b0b0b0',
        'hatch.color': c1,
        'image.cmap': 'hot',
        'patch.edgecolor': c1,
        'xtick.color': c1,
        'ytick.color': c1,
    }
    matplotlib.pyplot.rcParams.update(params)
