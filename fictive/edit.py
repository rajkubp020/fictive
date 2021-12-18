import matplotlib.pyplot as plt

def fix_zero(fig=None):
    if fig == None:
        axs = plt.gcf().get_axes()
    else:
        axs = fig.get_axes()

    def func(item):
        item.set_text(func_(item))
    def func_(item):
        x = item.get_text()
        y = x[14:-2]
        if y[0].isdigit():
            n = float(y)
        else:
            n = -float(y[1:])
        if n==0:
            return f'0'
        else:
            return x
    for ax in axs:
        # axes ticklabels
        ax.set_xticklabels([func_(item) for item in ax.get_xticklabels()])
        ax.set_yticklabels([func_(item) for item in ax.get_yticklabels()])


def fix_minus(fig=None, str_='\N{MINUS SIGN}'):
    if fig == None:
        axs = plt.gcf().get_axes()
    else:
        axs = fig.get_axes()

    def func(item):
        item.set_text(func_(item))

    def func_(item):
        x = item.get_text()
        if '-' in x:
            return f'{x}'.replace('-', str_)
        else:
            return x

    for ax in axs:
        # texts
        for item in ax.texts:
            func(item)
        # title
        func(ax.title)
        # axes labels
        func(ax.xaxis.label)
        func(ax.yaxis.label)
        # axes ticklabels
        ax.set_xticklabels([func_(item) for item in ax.get_xticklabels()])
        ax.set_yticklabels([func_(item) for item in ax.get_yticklabels()])
        # legend
        if ax.get_legend() != None:
            for item in ax.get_legend().get_texts():
                func(item)

def apply_font(axs, lst=['size', 'family', 'weight']):
    """
    Apply rcParams to current axs.
    Input Arguments:
        axs: axs handle
        lst: ['size', 'family', 'weight']
    """
    def func(item, s, f, w):
        if 'size' in lst:
            item.set_fontsize(s)
        if 'family' in lst:
            item.set_fontfamily(f)
        if 'weight' in lst:
            item.set_fontweight(w)
    for ax in axs:
        # texts
        for item in ax.texts:
            func(item, plt.rcParams['font.size'],
                 plt.rcParams['font.family'], plt.rcParams['font.weight'])
        # title
        func(ax.title, plt.rcParams['axes.titlesize'],
             plt.rcParams['font.family'], plt.rcParams['axes.titleweight'])
        # axes labels
        func(ax.xaxis.label, plt.rcParams['axes.labelsize'],
             plt.rcParams['font.family'], plt.rcParams['axes.labelweight'])
        func(ax.yaxis.label, plt.rcParams['axes.labelsize'],
             plt.rcParams['font.family'], plt.rcParams['axes.labelweight'])
        # axes ticklabels
        for item in ax.get_xticklabels():
            func(item, plt.rcParams['xtick.labelsize'],
                 plt.rcParams['font.family'], plt.rcParams['font.weight'])
        for item in ax.get_yticklabels():
            func(item, plt.rcParams['ytick.labelsize'],
                 plt.rcParams['font.family'], plt.rcParams['font.weight'])
        # legend
        if ax.get_legend() != None:
            for item in ax.get_legend().get_texts():
                func(item, plt.rcParams['legend.fontsize'],
                     plt.rcParams['font.family'], plt.rcParams['font.weight'])
