#####################################################
#probabilistic model
#
#this module implements the probabilistic model for
#attribution modeling
#####################################################

#---------------------------------------------
#import modules
#---------------------------------------------
import pandas as pd

#---------------------------------------------
#function to compute the influence of each channel
#---------------------------------------------
def prob_mod(data,user_conv):

    """
    input
        data: input dataframe that two columns:
            user_id, medium (ie. channel, line item)
        user_conv: dictionary. keys are the medium and their values
            if there is a conversion 1 or not 0

    output
        conv: dictionary. keys are user_id and values are 1 if conversion
        or 0 if non conversion
    """
    #probabilistic model 
    medium_unique = pd.Series.unique(data.iloc[:,1])

    #initializes the dictionaries that contains the number of positive (conversion)
    #and negative (non conversion) times a medium appears in customer journeys
    medium_p = {}
    medium_n = {}
    medium_p_aux = {}
    medium_n_aux = {}
    for i in medium_unique:
        medium_p[i] = 0
        medium_n[i] = 0
        medium_p_aux[i] = 0
        medium_n_aux[i] = 0

    data_shape = data.shape
    user_id_new = ''
    user_id_old = ''
    for i in range(data_shape[0]):
        user_id_new = data.iloc[i,0] 
        if user_id_new == user_id_old:
            if user_conv[data.iloc[i,0]] == 1:
                medium_p_aux[data.iloc[i,1]] = 1
            else:
                medium_p_aux[data.iloc[i,1]] = 1
        else:
            for j in medium_unique:
                medium_p[j] = medium_p[j]+medium_p_aux[j]
                medium_n[j] = medium_n[j]+medium_n_aux[j]
                medium_p_aux[j] = 0
                medium_n_aux[j] = 0

            user_id_old = user_id_new
    for i in medium_unique:
        medium_p[i] = medium_p[i]+medium_p_aux[i]
        medium_n[i] = medium_n[i]+medium_n_aux[i]

    return(medium_p, medium_n)


#---------------------------------------------
#function to create a dictionary with the information of the
#conversion of each user
#---------------------------------------------
def user_conversion(data):

    """
    input
        data: input dataframe with two columns:
            user_id, conversion or not

    output
        user_conv: output dictionary. keys are user_id and values 
        are 1 if conversion or 0 if non conversion
    """

    #unique user_id to create the dictionary
    user_unique = data.iloc[:,0].unique()

    #dictionary
    user_conv = {}
    for i in user_unique:
        user_conv[i] = 0
    nrow = data.shape[0]
    print(nrow)
    for i in range(nrow):
        if data.iloc[i,1] == 1:
            user_conv[data.iloc[i,0]] = 1
    return(user_conv)

#---------------------------------------------
#only runs the code if executed as main
#---------------------------------------------
if __name__== '__main__':
    print('Running this file as the main file does nothing')