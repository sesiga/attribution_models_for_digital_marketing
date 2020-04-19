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

#---------------------------------------------
#main program
#---------------------------------------------
def main():

    start_time = time.time()

    filepath = r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\criteo_attribution_dataset.tsv.gz'
    df = pd.read_csv(filepath, usecols=['timestamp','uid','campaign','conversion'], sep='\t', compression='gzip',nrows=5000)
    #df = pd.read_csv(filepath, sep='\t', compression='gzip')
    #pd.DataFrame.to_csv(df,r'C:\Users\sesig\Documents\master data science\tfm\criteo_attribution_dataset\criteo.csv',sep=',',index=False)

    probmod = probabilistic_model_v2.prob_mod(df.loc[:,['uid','campaign','conversion']])
    # print(probmod.iloc[1:10,:])
    # print(np.sum(probmod.iloc[:,1]))
    filepath_probmod = r'C:\Users\sesig\Documents\master data science\tfm\criteo_output_data\probabilistic_model.csv'
    pd.DataFrame.to_csv(probmod,path_or_buf=filepath_probmod,sep=',',index=False)
    print(time.time()-start_time)


#only runs the code if executed as main
if __name__== '__main__':
    main()