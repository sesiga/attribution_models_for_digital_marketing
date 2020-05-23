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

    d = {'medium':np.arange(1,13,1,dtype=np.int_),'contribution':np.zeros(12,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    prob_results = models.prob_mod(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(prob_results)

    last_results = models.LastTouchModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(last_results)

    linear_results = models.LinearModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(linear_results)

    # lr_results = models.LRmodel(data_lr.iloc[:,1:])
    # print(lr_results)

    first_results = models.FirstInteractionModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(first_results)

    bathtub_results = models.PositionBasedModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(bathtub_results)

    pos_decay_results = models.PositionDecayModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(pos_decay_results)

    medium_contribution['prob'] = prob_results['contribution']
    medium_contribution['last'] = last_results['contribution']
    medium_contribution['linear'] = linear_results['contribution']
    medium_contribution['first'] = first_results['contribution']
    medium_contribution['bathtub'] = bathtub_results['contribution']
    medium_contribution['pos_decay'] = pos_decay_results['contribution']

    path_contribution_out = r'C:\Users\sesig\Documents\master data science\tfm\results\contribution_simple_models.csv'
    pd.DataFrame.to_csv(medium_contribution,path_or_buf=path_contribution_out,sep=',',index=False)
    
    print(time.time()-start_time)


#only runs the code if executed as main
if __name__== '__main__':

    data, data_lr = data_config.data_in_r()
    main(data, data_lr)

    # data = data_config.data_in1()
    # main(data)