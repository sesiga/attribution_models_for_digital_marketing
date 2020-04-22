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
import numpy as np

#---------------------------------------------
#function to compute the influence of each channel
#---------------------------------------------
def prob_mod(data):

    """
    input
        data: input dataframe with three columns:
            user_id, medium (ie. channel, line item), conversion or not of each user

    output
        medium_contribution: dataframe with two columns:
            medium, contribution
    """
    medium_unique = pd.Series.unique(data.iloc[:,1])
    n_medium = len(medium_unique)

    medium_map = {}
    j = 0
    for i in medium_unique:
        medium_map[i] = j
        j += 1

    #numpy arrays to store the number of times of appereance of each channel
    medium_p = np.zeros((n_medium,n_medium),dtype=np.float_)
    medium_n = np.zeros((n_medium,n_medium),dtype=np.float_)
    medium_p_aux = np.zeros(n_medium,dtype=np.float_)
    medium_n_aux = np.zeros(n_medium,dtype=np.float_)

    data_shape = data.shape
    user_id_new = ''
    user_id_old = ''
    for i in range(data_shape[0]):
        user_id_new = data.iloc[i,0] 
        if user_id_new == user_id_old:
            if data.iloc[i,2]:
                medium_p_aux[medium_map[data.iloc[i,1]]] = 1.
            else:
                medium_n_aux[medium_map[data.iloc[i,1]]] = 1.
        else:
            medium_p += medium_p_aux
            medium_n += medium_n_aux
            medium_p_aux[:] = 0.
            medium_n_aux[:] = 0.
            if data.iloc[i,2]:
                medium_p_aux[medium_map[data.iloc[i,1]]] = 1.
            else:
                medium_n_aux[medium_map[data.iloc[i,1]]] = 1.
            user_id_old = user_id_new

    medium_p = medium_p/(medium_p+medium_n)
    medium_p_diag_sum = np.sum(np.diag(medium_p))
    d = {'medium':medium_unique,'contribution':np.zeros(n_medium,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)
    for i in range(n_medium):
        medium_contribution.iloc[i,1] =  medium_p[i,i]+(np.sum(medium_p[i,:])-medium_p_diag_sum-(n_medium-1)*medium_p[i,i])/(2.*(n_medium-1))

    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])
    
    return(medium_contribution)


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
    for i in range(nrow):
        if data.iloc[i,1] == 1:
            user_conv[data.iloc[i,0]] = 1
    return(user_conv)

#---------------------------------------------
#only runs the code if executed as main
#---------------------------------------------
if __name__== '__main__':
    print('Running this file as the main file does nothing')