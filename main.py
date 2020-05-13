#####################################################
#Main
#
#this file runs all the program
#####################################################

#---------------------------------------------
#import modules
#---------------------------------------------
import pandas as pd
import numpy as np
import probabilistic_model
import probabilistic_model_v2
import time
import data_processing
import data_processing_rpackage
import data_config
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import models


#---------------------------------------------
#main program
#---------------------------------------------
def main(data, data_lr):

    start_time = time.time()

    # prob_results = models.prob_mod(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(prob_results)

    # last_results = models.LastTouchModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(last_results)

    # linear_results = models.LinearModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(linear_results)

    lr_results = models.LRmodel(data_lr.iloc[:,1:])
    print(lr_results)

    print(time.time()-start_time)


#only runs the code if executed as main
if __name__== '__main__':

    data, data_lr = data_config.data_in_r()

    main(data, data_lr)