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

    # data_processing_rpackage.generate_ad_timestamp(data)
    
    channels = [23644447,16491630,73327,5061834,21016759,9100690,19602309,17710659,30535894,8980571,32368244,14235907]
    
    # for i in channels:

    #     x = data_processing.channel_ads_interarrival_times(data,i)

    #     plt.hist(x)
    #     plt.show()

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(data_processing.channel_ads_interarrival_times(data,32368244))
    # plt.xlabel('Time')
    # plt.ylabel('Count')
    # plt.subplot(2,1,2)
    # plt.hist(data_processing.channel_ads_interarrival_times(data,14235907))
    # plt.xlabel('Time')
    # plt.ylabel('Count')
    # plt.show()

    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, constrained_layout=True)
    # ax[0].hist(data_processing.channel_ads_interarrival_times(data,32368244), bins=20, range=(0,10))
    # ax[0].set_xlabel('Time (days)')
    # ax[0].set_ylabel('Count')
    # ax[1].hist(data_processing.channel_ads_interarrival_times(data,14235907), bins=20, range=(0,10))
    # ax[1].set_xlabel('Time (days)')
    # ax[1].set_ylabel('Count')
    # ax[2].hist(data_processing.channel_ads_interarrival_times(data,8980571), bins=20, range=(0,10))
    # ax[2].set_xlabel('Time (days)')
    # ax[2].set_ylabel('Count')
    # plt.show()

    data_processing.fit_gamma_dist(data)
    


    print(time.time()-start_time)


#only runs the code if executed as main
if __name__== '__main__':

    data = data_config.data_in1()

    main(data)