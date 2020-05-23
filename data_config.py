#####################################################
#data_config
#
#loads and exports data
#####################################################

#---------------------------------------------
#import modules
#---------------------------------------------
import pandas as pd

def data_in1():
    path1 = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\data_user_conversion.tsv.gz'
    data1 = pd.read_csv(path1, usecols=['timestamp','campaign','uid'], sep='\t', compression='gzip')
    path2 = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\data_user_non_conversion.tsv.gz'
    data2 = pd.read_csv(path2, usecols=['timestamp','campaign','uid'], sep='\t', compression='gzip')

    return(pd.concat([data1, data2]))

def data_out1(data_conversion, data_non_conversion, user_conversion):
    filepath_data_conversion = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\data_user_conversion.tsv.gz'
    pd.DataFrame.to_csv(data_conversion,path_or_buf=filepath_data_conversion,sep='\t',compression='gzip',index=False)
    # filepath_data_conversion = r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\data_user_conversion.csv'
    # pd.DataFrame.to_csv(data_conversion,path_or_buf=filepath_data_conversion,sep=',',index=False)

    filepath_data_non_conversion = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\data_user_non_conversion.tsv.gz'
    pd.DataFrame.to_csv(data_non_conversion,path_or_buf=filepath_data_non_conversion,sep='\t',compression='gzip',index=False)
    # filepath_data_non_conversion = r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\data_user_non_conversion.csv'
    # pd.DataFrame.to_csv(data_non_conversion,path_or_buf=filepath_data_non_conversion,sep=',',index=False)

    filepath_which_user_conversion = r'C:\Users\sesig\Documents\master data science\tfm\criteo_cleaned_data\which_user_conversion.tsv.gz'
    pd.DataFrame.to_csv(user_conversion,path_or_buf=filepath_which_user_conversion,sep='\t',compression='gzip',index=False)
    # filepath_which_user_conversion = r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\which_user_conversion.csv'
    # pd.DataFrame.to_csv(user_conversion,path_or_buf=filepath_which_user_conversion,sep=',',index=False)

    return(print('---data exported succesfully---'))

def data_in_r():
    path = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\r_dataset_timestamp.csv'
    data = pd.read_csv(path, sep=',')

    path_lr = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\data_for_lr.csv'
    data_lr = pd.read_csv(path_lr,sep=',')

    return data, data_lr

#only runs the code if executed as main
if __name__== '__main__':
    print('Running this file as the main file does nothing')
