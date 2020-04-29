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


#---------------------------------------------
#main program
#---------------------------------------------
def main(data):

    start_time = time.time()
    #data_conversion, data_non_conversion, user_conversion = data_processing.data_process(df)

    # channel = pd.read_csv(r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\channel_appereances.csv',sep=',',usecols=['campaign'],nrows=50)
    # channel_out = pd.DataFrame(columns=['campaign','gamma','exp'])

    # for i in channel['campaign']:

    #     x = data_processing.channel_ads_interarrival_times(data,i)

    #     #data_processing.plot_channel_distribution(x)

    #     gamma = data_processing.ks_test_gamma(x,stats.gamma,500)
    #     exp = data_processing.ks_test_exp(x,stats.expon,500)
    #     append = {'campaign':i, 'gamma':gamma, 'exp':exp}
    #     channel_out = pd.DataFrame.append(channel_out,append,ignore_index=True)

    # channel_out.to_csv(r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\channel_distribution.csv',sep=',')

    data_processing_rpackage.generate_ad_timestamp(data)


    print(time.time()-start_time)


#only runs the code if executed as main
if __name__== '__main__':

    data = data_config.data_in1()

    main(data)