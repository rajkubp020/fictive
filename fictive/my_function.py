#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:27:50 2019

@author: rajesh
"""



import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "/Users/rajesh/ownCloud/Shared/M3RG/Codes/my_python")
from log import reader
from plot import *



def my_plot(file = "log.lammps",save_png="save_.png",save_csv="save_.csv"):


    import numpy as np
    import pandas as pd
    #import matplotlib.pyplot as plt
    import sys

    sys.path.insert(0, "/Users/rajesh/ownCloud/Shared/M3RG/Codes/my_python")
    from log import reader
    from plot import *

    set_things()

    data=reader(file)
    data2=data['df'][2]
    data2.keys()
    t = data2['Temp']
    data1 = data2['Density']
    data2 = data2['PotEng']/23.06/2996
    a=50

    fig, [ax1] = panel(1,1,dpi=100,r_p=0.19,l_p=0.12,b_p=0.1,t_p=0.05)

    xlabel('Temp (K)')
    ylabel('Density g/cm3', color='b')
    #ax1.plot(t, data1, color='b')
    ax1.scatter(t, data1, color='b')

    ax2, ax1 = twinx(ax1)  # instantiate a second axes that shares the same x-axis

    ylabel('Enthalpy (Ev/atom)', color='r')  # we already handled the x-label with ax1
    #ax2.plot(t, data2, color='r')
    ax2.scatter(t, data2, color='r')


    fig.get_children()
    fig.get_children()[2].get_children()
    plt.savefig(save_png)

    plt.show()

    data4=data['df'][4]
    z=data4[-1:]
    np.savetxt('save_csv', z, delimiter=',')   # X is an array
#data4=data['df'][4]
#z = data4.iloc[100]
#
#np.savetxt('test.out', z, delimiter=',')
#data4[:-1]


#data4[:end-1,:]
#data4[-1 ,  :]
