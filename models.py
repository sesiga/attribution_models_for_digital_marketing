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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#---------------------------------------------
#probabilistic model
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
#last touch model
#---------------------------------------------
def LastTouchModel(data):
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

    d = {'medium':medium_unique,'contribution':np.zeros(n_medium,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    data_grouped = pd.DataFrame.groupby(data, by='uid')
    for name, group in data_grouped:
        if group.iloc[0,2] == 1:
            n_ads = group.shape[0]
            medium_loc = medium_map[group.iloc[n_ads-1,1]]
            medium_contribution.iloc[medium_loc,1] += 1
    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])

    return(medium_contribution)

#---------------------------------------------
#first interaction model
#---------------------------------------------
def FirstInteractionModel(data):
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

    d = {'medium':medium_unique,'contribution':np.zeros(n_medium,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    data_grouped = pd.DataFrame.groupby(data, by='uid')
    for name, group in data_grouped:
        if group.iloc[0,2] == 1:
            medium_loc = medium_map[group.iloc[0,1]]
            medium_contribution.iloc[medium_loc,1] += 1
    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])

    return(medium_contribution)

#---------------------------------------------
#position based model
#---------------------------------------------
def PositionBasedModel(data):
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
    medium_aux = {}
    j = 0
    for i in medium_unique:
        medium_map[i] = j
        medium_aux[i] = 0
        j += 1

    d = {'medium':medium_unique,'contribution':np.zeros(n_medium,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    data_grouped = pd.DataFrame.groupby(data, by='uid')
    for name, group in data_grouped:
        if group.iloc[0,2] == 1:
            n_ads = group.shape[0]
            medium_aux[group.iloc[0,1]] += 0.4 
            medium_aux[group.iloc[n_ads-1,1]] += 0.4 
            for i in range(1,n_ads-1):
                medium_aux[group.iloc[i,1]] += 0.2 / ( n_ads - 2 )
            for i in medium_aux:
                medium_loc = medium_map[i]
                medium_contribution.iloc[medium_loc,1] += medium_aux[i]
                medium_aux[i] = 0
    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])

    return(medium_contribution)

#---------------------------------------------
#position decay model
#---------------------------------------------
def PositionDecayModel(data):
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
    medium_aux = {}
    j = 0
    for i in medium_unique:
        medium_map[i] = j
        medium_aux[i] = 0
        j += 1

    d = {'medium':medium_unique,'contribution':np.zeros(n_medium,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    data_grouped = pd.DataFrame.groupby(data, by='uid')
    for name, group in data_grouped:
        if group.iloc[0,2] == 1:
            n_ads = group.shape[0]
            j = 1
            for i in range(n_ads):
                j += 1
            for i in range(n_ads):
                medium_aux[group.iloc[i,1]] += ( i + 1 ) / ( n_ads * j )
            for i in medium_aux:
                medium_loc = medium_map[i]
                medium_contribution.iloc[medium_loc,1] += medium_aux[i]
                medium_aux[i] = 0
    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])

    return(medium_contribution)

#---------------------------------------------
#linear touch model
#---------------------------------------------
def LinearModel(data):
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
    medium_aux = {}
    j = 0
    for i in medium_unique:
        medium_map[i] = j
        medium_aux[i] = 0
        j += 1

    d = {'medium':medium_unique,'contribution':np.zeros(n_medium,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    data_grouped = pd.DataFrame.groupby(data, by='uid')
    for name, group in data_grouped:
        if group.iloc[0,2] == 1:
            n_ads = group.shape[0]
            for i in range(n_ads):
                medium_aux[group.iloc[i,1]] += 1 / n_ads
            for i in medium_aux:
                medium_loc = medium_map[i]
                medium_contribution.iloc[medium_loc,1] += medium_aux[i]
                medium_aux[i] = 0
    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])

    return(medium_contribution)

#---------------------------------------------
#logistic regression model
#---------------------------------------------
def LRmodel(data):
    """
    input
        data: input dataframe with columns:
            user_id, medium (ie. channel, line item), conversion or not of each user

    output
        medium_contribution: dataframe with two columns:
            medium, contribution
    """

    n_iter = 1000
    test_size = 0.25
    p_obs = 0.3
    p_var = 0.4

    ch = np.arange(0,12,1,dtype=np.int_)
    d = {'medium':ch, 'contribution':np.zeros(12,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)
    
    for i in range(n_iter):
        data_sample = data.sample(frac=p_obs, random_state=i, axis=0)
        ch_sample = data_sample.iloc[:,0:11].sample(frac=p_var, random_state=i, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(ch_sample.values, data_sample['conversion'], test_size=test_size, random_state=i, stratify=data_sample['conversion'])
        lr = LogisticRegression(penalty='l2', C=10000.,  fit_intercept=True, solver='newton-cg')
        lr.fit(x_train, y_train)
        lr_pred = lr.predict(x_test)
        lr_coef = lr.coef_

        k = 0
        for j in ch_sample.columns:
            medium_contribution.iloc[int(j),1] = lr_coef[0,k]
            k += 1

    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])

    return(medium_contribution)
    
#---------------------------------------------
#transform data set to matrix format before using the data in logistic regression model
#---------------------------------------------
def TransformDataToLRmodel(data):
    """
    input
        data: input dataframe with columns:
            user_id, medium (ie. channel, line item), conversion or not of each user

    output
        data_out: dataframe with columns:
            user_id, medium (ie. channel, line item), conversion or not of each user
    """
    data_grouped = pd.DataFrame.groupby(data, by='uid')
    user_unique = pd.Series.unique(data.iloc[:,0])
    n_user = len(user_unique)
    ch_unique = pd.Series.unique(data.iloc[:,1])
    n_ch = len(ch_unique)
    data_out = pd.DataFrame(data={'uid':user_unique})
    for i in ch_unique:
        data_out[i] = np.zeros(n_user, dtype=np.int_)
    data_out['conversion'] = np.zeros(n_user, dtype=np.int_)
    j = 0
    for name, group in data_grouped:
        data_out.loc[name,'conversion'] = group.iloc[0,2]
        for i in group.index:
            ch = data.iloc[i,1]
            data_out.loc[j,ch] = 1
        j += 1

    path = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\data_for_lr.csv'
    pd.DataFrame.to_csv(data_out,path_or_buf=path,sep=',',index=False)


#---------------------------------------------
#only runs the code if executed as main
#---------------------------------------------
if __name__== '__main__':
    print('Running this file as the main file does nothing')