#####################################################
#logistic regression model
#
#this module implements the logistic regession model for
#attribution modeling
#####################################################

#---------------------------------------------
#import modules
#---------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

#---------------------------------------------
#function to compute the influence of each channel
#---------------------------------------------
def lr_mod(x,y):

    """
    input
        x: numpy array matrix. each row is a user and each column is a channel. If channel i is included, then its value is 1, if not included then 0
        y: numpy array. if conversion of the user then 1, if not 0
    output

    """

    lr = LogisticRegression(penalty='l2',C=1.,fit_intercept=True,solver='sag')
    lr.fit(x,y)
    return(lr.get_params, lr.score)


#---------------------------------------------
#only runs the code if executed as main
#---------------------------------------------
if __name__== '__main__':
    print('Running this file as the main file does nothing')