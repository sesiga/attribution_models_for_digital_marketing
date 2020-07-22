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
import data_processing

#---------------------------------------------
#function
#---------------------------------------------
def process_data_r():
    
    path_r_data = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset\r_data.csv'
    data = pd.read_csv(filepath_or_buffer=path_r_data, sep=',')
    
    data_processed = pd.DataFrame(columns=['uid','campaign','total_conversion','total_null'])
    
    for i in data.index:
        cj = data.iloc[i,0]
        cj_split = str.split(cj, '>')
        for j in cj_split:
            data_to_append = {'uid':i, 'campaign':j, 'total_conversion':data.iloc[i,1], 'total_null':data.iloc[i,3]}
            data_processed = pd.DataFrame.append(data_processed, data_to_append, ignore_index=True)
            
    path_processed = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\r_dataset_cleaned.csv'
    pd.DataFrame.to_csv(data_processed,path_or_buf=path_processed,sep=',',index=False)
    
    return(print('process_data_r finished execution'))
    
def map_col():
    
    path_processed = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\r_dataset_cleaned.csv'
    data = pd.read_csv(filepath_or_buffer=path_processed, sep=',')
    
    for i in data.index:
        data.loc[i,'campaign'] = str.strip(data.loc[i,'campaign'])
        
    campaign = pd.Series.unique(data['campaign'])
    
    campaign_map = {}
    j = 0
    for i in campaign:
        campaign_map[i] = j
        j = j+1
        
    data['campaign'] = data['campaign'].map(campaign_map)
    
    path_processed_map = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\r_dataset_cleaned_map.csv'
    pd.DataFrame.to_csv(data,path_or_buf=path_processed_map,sep=',',index=False)
    
    return(print('process_data_r finished execution'))
        
def split_users():
    
    path_in = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\r_dataset_cleaned_map.csv'
    data = pd.read_csv(filepath_or_buffer=path_in, sep=',')
    
    data_split = pd.DataFrame(columns=['uid','campaign','conversion'])
    
    data_grouped = pd.DataFrame.groupby(data,by='uid')
    k = 0
    for name, group in data_grouped:
        for i in range(group.iloc[0,2]):
            #data_to_append = {'uid':group.loc[:,'uid'], 'campaign':group.loc[:,'campaign'], 'conversion':np.ones((len(group),), dtype=int)}
            data_to_append = pd.DataFrame(columns=['uid','campaign','conversion'])
            data_to_append['campaign'] = group.loc[:,'campaign']
            data_to_append['uid'] = k
            data_to_append['conversion'] = 1
            data_split = pd.DataFrame.append(data_split, data_to_append, ignore_index=True)
            k = k+1
        for i in range(group.iloc[0,3]):
            #data_to_append = {'uid':group['uid'], 'campaign':group['campaign'], 'conversion':np.zeros((len(group),), dtype=int)}
            data_to_append = pd.DataFrame(columns=['uid','campaign','conversion'])
            data_to_append['campaign'] = group.loc[:,'campaign']
            data_to_append['uid'] = k
            data_to_append['conversion'] = 0
            data_split = pd.DataFrame.append(data_split, data_to_append, ignore_index=True)
            k = k+1
            
    path_out = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\r_dataset_cleaned_map_split_users.csv'
    pd.DataFrame.to_csv(data_split,path_or_buf=path_out,sep=',',index=False)    

def generate_ad_timestamp(data_criteo):
    data_r_path_in = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\r_dataset_cleaned_map_split_users.csv'
    data_r = pd.read_csv(filepath_or_buffer=data_r_path_in, sep=',')
    
    channels = [23644447,16491630,73327,5061834,21016759,9100690,19602309,17710659,30535894,8980571,32368244,14235907]
    criteo_map_to_data_r = {}
    data_r_map_to_criteo = {}
    channel_params = {}

    j = 0
    for i in channels:

        x = data_processing.channel_ads_interarrival_times(data_criteo,i)

        channel_params[i] = stats.gamma.fit(x)
        criteo_map_to_data_r[i] = j
        data_r_map_to_criteo[j] = i
        j += 1

    data_r['timestamp'] = 0
    data_r_grouped = pd.DataFrame.groupby(data_r,by='uid')
    data_r_uid = {}
    data_r_unique_users = data_r['uid'].unique()
    for name, group in data_r_grouped:
        data_r_uid[name] = len(group)

    k = 0
    for i in data_r_unique_users:
        print(i)
        data_r.loc[k,'timestamp'] = 0
        k += 1
        for j in range(data_r_uid[i]-1):
            data_r_channel = data_r.loc[k,'campaign']
            criteo_channel = data_r_map_to_criteo[data_r_channel]
            a = channel_params[criteo_channel][0]
            loc = channel_params[criteo_channel][1]
            scale = channel_params[criteo_channel][2]
            data_r.loc[k,'timestamp'] = data_r.loc[k-1,'timestamp'] + stats.gamma.rvs(a, loc=loc, scale=scale, size=1, random_state=k)
            k += 1

    path_out = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\r_dataset_timestamp.csv'
    pd.DataFrame.to_csv(data_r,path_or_buf=path_out,sep=',',index=False) 

    return(print('---generate_ad_timestamp---'))
    
def generate_conv_timestamp():
    #add timestamp of conversion for each user
    path = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\data_all_1u.csv'
    data = pd.read_csv(filepath_or_buffer=path, sep=',')
    data_grouped = pd.groupby(data, by='uid')
    nuser = pd.Series.nunique(data['uid'])
    x = pd.DataFrame(data={'uid':np.arange(nuser,dtype=np.int_), 'tconv':np.zeros(nuser,dtype=np.float_)})

    path_params = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\gamma_dist_params.csv'
    channel_params = pd.read_csv(filepath_or_buffer=path_params, sep=',')

    i = 0
    for name, group in data_grouped:
        if group.iloc[0,2] == 1:
            ch = group.iloc[-1,1]
            a = channel_params.loc[ch,'shape parameter']
            loc = channel_params.loc[ch,'location parameter']
            scale = channel_params.loc[ch,'scale parameter']
            x.loc[i,'tconv'] = group.iloc[-1,3] + stats.gamma.rvs(a, loc=loc, scale=scale, size=1, random_state=i)
            i += 1
        else:
            x.loc[i,'tconv'] = group.iloc[-1,3] + 15
            i += 1

    path_out = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\r_dataset_tconv.csv'
    pd.DataFrame.to_csv(x,path_or_buf=path_out,sep=',',index=False)

#---------------------------------------------
#creates pandas dataframe from the original data
#---------------------------------------------

#only runs the code if executed as main
if __name__== '__main__':
    
    import pandas as pd
    import numpy as np
    
    #process_data_r()
    
    #map_col()
    
    split_users()
    
    


    

    
    
    
    
    
    
    
    
    
    
    