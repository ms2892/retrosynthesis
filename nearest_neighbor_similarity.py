import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from glob import glob
import json


def get_plot(files:list,np_files:list):

    x_s = []
    y_s = []
    for i in files:
        try:
            with open(i,'r') as f:
                dct = json.load(f)
            x = [int(j)/2 for j in dct.keys()]
            y = [1-float(j) for j in dct[str(x[0]).split(',')]]
            x = x*len(y)
            x_s += x 
            y_s += y 
        except:
            pass


    for i in np_files:
        try:
            points = np.load(i)

            x = [int(j)//2 for j in points[:,0]]
            y = list(points[:,1])
            y = [1-j for j in y]
            x_s +=x 
            y_s += y
        except:
            pass

    dct = {'depth':x_s,'value':y_s}

    data = pd.DataFrame.from_dict(dct)

    sns.boxplot(x='depth',y='value',data=dct)

    plt.show()