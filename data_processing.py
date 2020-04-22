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


        # for _ in range(len(group)):
        #     if group.iloc[0,3] == 1:
        #         conv = 1
        # which_user_conversion = pd.DataFrame.append(which_user_conversion, {'uid':name, 'conversion':conv}, ignore_index=True)
        # if conv:
        #     data_conversion = pd.DataFrame.append(data_conversion,group, ignore_index=True)
        # else:
        #     data_non_conversion = pd.DataFrame.append(data_non_conversion,group, ignore_index=True)


    return(data_conversion, data_non_conversion, which_user_conversion)

#only runs the code if executed as main
if __name__== '__main__':
    print('Running this file as the main file does nothing')












