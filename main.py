#####################################################
#Main
#
#this file runs all the program
#####################################################

#---------------------------------------------
#import modules
#---------------------------------------------
import pandas as pd
import probabilistic_model

#---------------------------------------------
#main program
#---------------------------------------------
import pandas as pd
def main():
    print('hello world')

    filepath1 = r'C:\Users\sesig\Documents\master data science\tfm\Colaboracion_C3_datos\hash_grupo_0.csv'
    cols1 = ['user_id','action_date','action_type','site_id','creative_id','conversion_action','conversion_id','keyword']
    cols1_dtype = {'user_id':'str','action_type':'str','site_id':'str','creative_id':'str','conversion_action':'str','conversion_id':'str','keyword':'str'}
    file1 = pd.read_csv(filepath1,sep=',',usecols=cols1,nrows=5000,dtype=cols1_dtype)
    print(file1.loc[0,['action_date','user_id']])
    print('---------------')
    print(file1.loc[0,cols1])
    print('---------------')

    file1['conversion'] = 0
    file1_shape = file1.shape

    for i in range(file1_shape[0]):
        if file1.loc[i,'action_type'] == 'conversion':
            file1.loc[i,'conversion'] = 1

    user_conv = probabilistic_model.user_conversion(file1.loc[:,['user_id','conversion']])

    file1 = pd.DataFrame.rename(file1,mapper={'site_id':'medium'},axis=1)

    print('---------------')
    print(file1.loc[0,:])
    print('---------------')

    medium_p, medium_n = probabilistic_model.prob_mod(file1.loc[:,['user_id','medium']],user_conv)

    print('---------------')
    print(medium_p['1233_171_'])
    print(medium_n['1233_171_'])
    print('---------------')

    print(user_conv['695cccad7341f61dffaedaf66b61046633030c25'])


#only runs the code if executed as main
if __name__== '__main__':
    main()