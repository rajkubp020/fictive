import sys
import numpy as np
import pandas as pd
import sys
import os
#!{sys.executable} -m pip install git+https://github.com/rajkubp020/fictive.git
#%matplotlib inline

import fictive
from fictive import *

from sklearn import linear_model
from sklearn import linear_model
x=pd.read_csv('./cc.csv')
x.columns.tolist()

fig, [ax] = panel(1,1)
xlabel(r'Size of ring  ')
ylabel('Normalized frequency ')
#beautify_leg(ax)
plt.plot(x['Unnamed: 0'],x[' Number'],'o-b', lw=2.5,label='Pristine 12B',ms=12)
plt.legend(fontsize=25)
legend_on(ax=ax)
plt.legend(loc=3)
plt.savefig('./fictive_.png'.format(83),dpi=300,bbox_inches='tight')
plt.show()

name = 'Suresh'
