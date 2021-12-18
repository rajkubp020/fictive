import matplotlib.font_manager as font_manager
import matplotlib
import os

def add_custom_fonts(loc="~/.myfonts"):
    full_path = os.path.expanduser(loc)
    font_dirs = [full_path, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    font_list = font_manager.createFontList(font_files)
    font_manager.fontManager.ttflist.extend(font_list)

def set_font_size(size=20):
    params = {
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'axes.labelsize': size,
        'axes.titlesize': size,
        'font.size': size,
        'legend.title_fontsize': size,
        'legend.fontsize': size,
        'figure.titlesize': size,
    }
    matplotlib.pyplot.rcParams.update(params)

def set_font_family(family='Arial'):
    params = {
        'font.family': family,
    }
    matplotlib.pyplot.rcParams.update(params)


def set_font_weight(weight='bold'):
    params = {
        'font.weight': weight,
        'axes.labelweight': weight,
        'axes.titleweight': weight,
    }
    matplotlib.pyplot.rcParams.update(params)
