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
# import matplotlib.pyplot as plt
import models



#---------------------------------------------
#main program
#---------------------------------------------
def main(data,data_lr,data_1u):

    start_time = time.time()

    # d = {'medium':np.arange(0,12,1,dtype=np.int_)}
    # medium_contribution = pd.DataFrame(data=d)

    # prob_results = models.prob_mod(data.loc[:,['uid', 'campaign', 'conversion']])
    # # print(prob_results)
    # prob_results = models.prob_mod(data_1u.loc[:,['uid', 'campaign', 'conversion']])
    # print(prob_results)

    # last_results = models.LastTouchModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(last_results)
    # last_results = models.LastTouchModel(data_1u.loc[:,['uid', 'campaign', 'conversion']])
    # print(last_results)

    # linear_results = models.LinearModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(linear_results)
    # linear_results = models.LinearModel(data_1u.loc[:,['uid', 'campaign', 'conversion']])
    # print(linear_results)

    # linear_same_results = models.LinearModelSame(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(linear_same_results)
    # linear_same_results = models.LinearModelSame(data_1u.loc[:,['uid', 'campaign', 'conversion']])
    # print(linear_same_results)

    # lr_optC = models.LRmodel(data_lr)

    # lr_results = models.LRmodel(data_lr.iloc[:,1:])
    # print(lr_results)
    # lr_results = models.LRmodel(data_lr.iloc[:,1:])
    # print(lr_results)

    # first_results = models.FirstInteractionModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(first_results)
    # first_results = models.FirstInteractionModel(data_1u.loc[:,['uid', 'campaign', 'conversion']])
    # print(first_results)

    # bathtub_results = models.PositionBasedModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(bathtub_results)
    # bathtub_results = models.PositionBasedModel(data_1u.loc[:,['uid', 'campaign', 'conversion']])
    # print(bathtub_results)

    # pos_decay_results = models.PositionDecayModel(data.loc[:,['uid', 'campaign', 'conversion']])
    # print(pos_decay_results)
    # pos_decay_results = models.PositionDecayModel(data_1u.loc[:,['uid', 'campaign', 'conversion']])
    # print(pos_decay_results)

    # time_decay_results = models.TimeDecayModel(data.loc[:,['uid', 'campaign', 'conversion', 'timestamp']])
    # print(time_decay_results)
    # time_decay_results = models.TimeDecayModel(data_1u.loc[:,['uid', 'campaign', 'conversion', 'timestamp']])
    # print(time_decay_results)

    # models.SurvivalModel()
    # adhazard_results = models.SurvivalModelCont()
    # print(adhazard_results)

    # medium_contribution['prob'] = prob_results['contribution']
    # medium_contribution['last'] = last_results['contribution']
    # medium_contribution['linear'] = linear_results['contribution']
    # medium_contribution['linear_same'] = linear_same_results['contribution']
    # medium_contribution['first'] = first_results['contribution']
    # medium_contribution['bathtub'] = bathtub_results['contribution']
    # medium_contribution['pos_decay'] = pos_decay_results['contribution']
    # medium_contribution['time_decay'] = time_decay_results['contribution']
    # medium_contribution['lr'] = lr_results['contribution']

    # path_contribution_out = r'C:\Users\sesig\Documents\master data science\tfm\results\contribution_models_1u.csv'
    # pd.DataFrame.to_csv(medium_contribution,path_or_buf=path_contribution_out,sep=',',index=False)

    # data_processing_rpackage.generate_conv_timestamp()

    # x = data_config.contribution_in()
    # models.contribution(x,0)

    # x = data_config.contribution_1u_in()
    # models.contribution(x,0)
    # models.contribution2(x,0)

    # data_processing.plot_interarrival_times()
    data_processing.plot_density_interarrival_times()
    # data_processing.channel_ads_interarrival_individual_times()

    # models.TransformDataToLRmodel_NoRepetition(data_all, data_agr)
    # models.TransformDataToLRmodel(data)
    # models.TransformDataToLRmodel(data_1u)
    
    print(time.time()-start_time)


#only runs the code if executed as main
if __name__== '__main__':

    data, data_lr, data_1u = data_config.data_in_r()
    main(data,data_lr,data_1u)

    # data_all, data_agr = data_config.data_in_r2()
    # main(data_all, data_agr)

    # data = data_config.data_in1()
    # main(data)