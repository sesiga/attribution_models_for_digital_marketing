#####################################################
#survival model
#
#this module implements the survival theory based model for
#attribution modeling
#####################################################

#---------------------------------------------
#import modules
#---------------------------------------------
import pandas as pd
import numpy as np
from scipy.optimize import minimize

#---------------------------------------------
#function to minimize
#---------------------------------------------

def objective_f(w,data):
    """
    input
        w: np.ndarray with the variables
        data: pd.dataframe with 4 columns
            user_id, timestamp, campaign, conversion, conversion_timestamp
    """

    #log likelihood function
    data_shape = data.shape
    user_id_new = ''
    user_id_old = ''
    for i in range(data_shape[0]): #for each user
        user_id_new = data.iloc[i,0]
        




#---------------------------------------------
#only runs the code if executed as main
#---------------------------------------------
if __name__== '__main__':
    print('Running this file as the main file does nothing')