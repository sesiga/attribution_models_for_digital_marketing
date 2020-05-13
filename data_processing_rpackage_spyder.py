import numpy as np
import pandas as pd

a = np.arange(0,11,1,dtype=np.int_)
b = np.zeros(12,dtype=np.float_)

d = pd.DataFrame(data={'a':np.arange(0,5,1,dtype=np.int_), 0:np.arange(1,6,1,dtype=np.int_), 1:np.arange(2,7,1,dtype=np.int_)})
print(d.columns)
ch = np.arange(0,12,1,dtype=np.int_)
print(d[[0,1]])