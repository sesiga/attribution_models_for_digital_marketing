#---------------------------------------------
#only runs the code if executed as main
#---------------------------------------------
if __name__== '__main__':
    
    import numpy as np
    import pandas as pd

    x = np.array([1.0,2.0,3.0,4.0])
    x1 = 1.
    p = 1./(1+np.exp(-x[0]*x1))
    b = x[1]*x[2]
    