#####################################################
#data_processing
#
#this pre process the data to be used in the algorithms
#of the main program
#####################################################

#---------------------------------------------
#import modules
#---------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

#----------------------------------
#load data
#----------------------------------
def data_process(data):

    #pass timestamp from hours to seconds
    data.iloc[:,0] = pd.Series.apply(data.iloc[:,0],lambda x: x/86400)
    data.iloc[:,4] = pd.Series.apply(data.iloc[:,4],lambda x: x/86400)
    data.iloc[0:20,:]

    df_groupby_uid = pd.DataFrame.groupby(data,by='uid')

    which_user_conversion = pd.DataFrame(data={'uid':[], 'conversion':[]}, dtype=np.int_)
    data_conversion = pd.DataFrame(columns=['timestamp','uid','campaign','conversion','conversion_timestamp'])
    data_non_conversion = pd.DataFrame(columns=['timestamp','uid','campaign','conversion','conversion_timestamp'])
    for name, group in df_groupby_uid:
        if group.iloc[0,3]:
            which_user_conversion = pd.DataFrame.append(which_user_conversion, {'uid':name, 'conversion':1}, ignore_index=True)
            data_conversion = pd.DataFrame.append(data_conversion,group, ignore_index=True)
        else:
            which_user_conversion = pd.DataFrame.append(which_user_conversion, {'uid':name, 'conversion':0}, ignore_index=True)
            data_non_conversion = pd.DataFrame.append(data_non_conversion,group, ignore_index=True)

    return(data_conversion, data_non_conversion, which_user_conversion)

def channel_ads_interarrival_times(data, group):

    channel = data.groupby(['campaign'],sort=False).get_group(group)
    channel_uid = channel.groupby(['uid'])
    x = pd.DataFrame(columns=['timestamp'])
    for name, group in channel_uid:
        group.sort_values(by=['timestamp'])
        group_index = group.index
        original_timestamp_1 = group.loc[group_index[0],'timestamp']
        if len(group) > 1:
            for i in group_index[1:]:
                original_timestamp = group.loc[i,'timestamp']
                aux = group.loc[i,'timestamp']-original_timestamp_1
                x = pd.DataFrame.append(x,{'timestamp':aux}, ignore_index=True)
                original_timestamp_1 = original_timestamp

    x1 = np.array(x['timestamp'].values, dtype=np.float_)

    return(x1)

def plot_channel_distribution(x):

    print(sm.stats.diagnostic.kstest_exponential(x,dist='exp'))
    sm.qqplot(x,dist=stats.gamma,fit=True,line='45')
    plt.show()

    print(stats.expon.fit(x))
    params = stats.expon.fit(x)
    exp_sim = stats.expon.rvs(loc=params[0],scale=params[1],size=len(x))
    exp_sim_params = stats.expon.fit(exp_sim)
    print(stats.kstest(x,'expon',exp_sim_params))
    print(stats.anderson(exp_sim,'expon'))
    stats.probplot(exp_sim_params,dist='expon',sparams=params,plot=plt)
    plt.show()
    # print(stats.expon.fit(x))
    plt.hist(x)
    plt.show()
    stats.gaussian_kde(x)
    plt.show()

def ks_test_gamma(x,dist,n_sample):
    x_params = dist.fit(x)
    x_ks_test = stats.kstest(x,dist.cdf,x_params)[0]

    n_obs = len(x)
    statistic = np.zeros(n_sample)
    for i in range(n_sample):
        x1 = dist.rvs(x_params[0],size=n_obs)
        x1_params = dist.fit(x1)
        statistic[i] = stats.kstest(x1,dist.cdf,x1_params)[0]

    p_value = len(statistic[statistic > x_ks_test])/n_sample

    print('p-value gamma')
    print(p_value)

    return(p_value)

def ks_test_exp(x,dist,n_sample):
    x_params = dist.fit(x)
    x_ks_test = stats.kstest(x,dist.cdf,x_params)[0]

    n_obs = len(x)
    statistic = np.zeros(n_sample)
    for i in range(n_sample):
        x1 = dist.rvs(size=n_obs)
        x1_params = dist.fit(x1)
        statistic[i] = stats.kstest(x1,dist.cdf,x1_params)[0]

    p_value = len(statistic[statistic > x_ks_test])/n_sample

    print('p-value exponential')
    print(p_value)

    return(p_value)

def fit_gamma_dist(data):
    
    channels = [23644447,16491630,73327,5061834,21016759,9100690,19602309,17710659,30535894,8980571,32368244,14235907]

    j = 0
    df = pd.DataFrame(columns=['channel', 'p-value', 'shape parameter', 'scale parameter', 'location parameter'])
    df['channel'] = channels
    df['p-value'] = 0
    for i in channels:

        x = channel_ads_interarrival_times(data,i)

        params = stats.gamma.fit(x)

        df.loc[j,'shape parameter'] = params[0]
        df.loc[j, 'location parameter'] = params[1]
        df.loc[j, 'scale parameter'] = params[2]
        j += 1

    path = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\gamma_dist_params.csv'
    pd.DataFrame.to_csv(df,path_or_buf=path,sep=',',index=False)

def channel_ads_distribution(data,channels):
    channel = pd.read_csv(r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\channel_appereances.csv',sep=',',usecols=['campaign'],nrows=50)
    channel_out = pd.DataFrame(columns=['campaign','gamma','exp'])

    for i in channel['campaign']:

        x = channel_ads_interarrival_times(data,i)

        #plot_channel_distribution(x)

        gamma = ks_test_gamma(x,stats.gamma,500)
        exp = ks_test_exp(x,stats.expon,500)
        append = {'campaign':i, 'gamma':gamma, 'exp':exp}
        channel_out = pd.DataFrame.append(channel_out,append,ignore_index=True)


def n_of_channel_appereances(data):
    channel_length = pd.DataFrame(columns=['campaign','number'])
    data2 = data.groupby(['campaign'], sort=False)
    for name, group in data2:
        channel_length = pd.DataFrame.append(channel_length,{'campaign':name, 'number':len(group)},ignore_index=True)
    path = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\channel_appereances.csv'
    pd.DataFrame.to_csv(channel_length,path_or_buf=path,sep=',',index=False)

#only runs the code if executed as main
if __name__== '__main__':
    print('Running this file as the main file does nothing')












