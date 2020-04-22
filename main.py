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

#---------------------------------------------
#main program
#---------------------------------------------
def main():

    start_time = time.time()

    # filepath = r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\criteo_attribution_dataset.tsv.gz'
    filepath_ordered = r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\criteo_ordered_dataset.tsv.gz'
    columns = ['timestamp','uid','campaign','conversion','conversion_timestamp']
    # df = pd.read_csv(filepath, usecols=columns, sep='\t', compression='gzip',nrows=5000)
    # df = pd.read_csv(filepath, usecols=columns, sep='\t', compression='gzip')
    # df = pd.read_csv(filepath_ordered, usecols=columns, sep='\t', compression='gzip',nrows=30000)
    df = pd.read_csv(filepath_ordered, usecols=columns, sep='\t', compression='gzip')
    #df = pd.read_csv(filepath, sep='\t', compression='gzip')
    #pd.DataFrame.to_csv(df,r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\criteo.csv',sep=',',index=False)

    # pd.DataFrame.sort_values(df,by=['uid','timestamp'],inplace=True)

    # filepath_out = r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\criteo_ordered_dataset.tsv.gz'
    # pd.DataFrame.to_csv(df,path_or_buf=filepath_out,sep='\t',compression='gzip',index=False)
    #pd.DataFrame.to_csv(df,path_or_buf=filepath_out,sep=',',index=False)

    data_conversion, data_non_conversion, user_conversion = data_processing.data_process(df)

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


    # probmod = probabilistic_model_v2.prob_mod(df.loc[:,['uid','campaign','conversion']])
    # filepath_probmod = r'C:\Users\sesig\Documents\master data science\tfm\criteo_output_data\probabilistic_model.csv'
    # pd.DataFrame.to_csv(probmod,path_or_buf=filepath_probmod,sep=',',index=False)
    print(time.time()-start_time)


#only runs the code if executed as main
if __name__== '__main__':
    main()