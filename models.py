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
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

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
#linear same touch model
#---------------------------------------------
def LinearModelSame(data):
    """
    Assigns credit equally to all channels visited no matter the number of touchpoints to each channel.
    Each channel is counted just once
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
            n_channels = pd.Series.nunique(group.iloc[:,1])
            for i in range(n_ads):
                medium_aux[group.iloc[i,1]] = 1 / n_channels
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
    threshold = 0.85

    ch = np.arange(0,12,1,dtype=np.int_)
    d = {'medium':ch, 'contribution':np.zeros(12,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:13].values, data['conversion'], \
        test_size=test_size, random_state=1, stratify=data['conversion'])
    lr = LogisticRegression(penalty='l2', C=0.001,  fit_intercept=False, solver='lbfgs')
    #c=0.1,1000
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    beta = lr.coef_
    y_pred2 = np.copy(y_pred)
    y_prob = lr.predict_proba(x_test)

    threshold = np.arange(0.2,0.4,0.01)
    for j in threshold:
        for i in range(len(y_pred)):
            if y_prob[i,1] > j:
                y_pred2[i] = 1
            else:
                y_pred2[i] = 0

    for i in range(len(medium_contribution)):
    # for i in range(6):
        medium_contribution.loc[i,'contribution'] = beta[0,i]
    # medium_max = max(beta[0,:6]) + 0.01
    # medium_min = min(beta[0,:6]) - 0.01
    medium_max = max(beta[0,:]) + 0.01
    medium_min = min(beta[0,:]) - 0.01

    print(medium_contribution)

    for i in range(len(medium_contribution)):
    # for i in range(6):
        medium_contribution.loc[i,'contribution'] = np.exp(beta[0,i]) / ( 1 + np.exp(beta[0,i]) )
    # medium_max = max(beta[0,:6]) + 0.01
    # medium_min = min(beta[0,:6]) - 0.01
    medium_max = max(beta[0,:]) + 0.01
    medium_min = min(beta[0,:]) - 0.01

    print(medium_contribution)

    # for i in range(len(medium_contribution)):
    # # for i in range(6):
    #     medium_contribution.iloc[i,1] = ( medium_contribution.iloc[i,1] - medium_min ) / ( medium_max - medium_min )

    # print(medium_contribution)

    medium_sum = np.sum(medium_contribution.iloc[:,1])
    for i in range(len(medium_contribution)):
        medium_contribution.iloc[i,1] /= medium_sum

    print(medium_contribution)

#---------------------------------------------
#logistic regression model
#---------------------------------------------
def LRmodel_optC(data):
    """
    function to optimize the value of the regularization term before obtaining the final model

    input
        data: input dataframe with columns:
            user_id, medium (ie. channel, line item), conversion or not of each user

    output
        medium_contribution: dataframe with two columns:
            medium, contribution
    """

    n_iter = 1000
    test_size = 0.25
    p_obs = 0.1
    p_var = 0.4
    regularization = [1000]
    # regularization = [1000]

    ch = np.arange(0,12,1,dtype=np.int_)
    d = {'medium':ch, 'contribution':np.zeros(12,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)
    table = np.zeros((2,2),dtype=np.float_)
    table2 = np.zeros((2,2),dtype=np.float_)
    coef = np.zeros((len(regularization),12),dtype=np.float_)

    j = 0
    threshold = 0.85
    for c in regularization:
        for i in range(n_iter):
            data_sample = data.sample(frac=p_obs, random_state=i, axis=0)
            # ch_sample = data_sample.iloc[:,1:13].sample(frac=p_var, random_state=i, axis=1)

            # x_train, x_test, y_train, y_test = train_test_split(ch_sample.values, data_sample['conversion'], \
            #     test_size=test_size, random_state=i, stratify=data_sample['conversion'])
            x_train, x_test, y_train, y_test = train_test_split(data_sample.iloc[:,1:13].values, data_sample['conversion'], \
                test_size=test_size, random_state=i, stratify=data_sample['conversion'])
            lr = LogisticRegression(penalty='l2', C=c,  fit_intercept=False, solver='lbfgs')
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)
            y_pred2 = np.copy(y_pred)
            y_prob = lr.predict_proba(x_test)
            for k in range(len(y_pred)):
                if y_prob[k,0] > threshold:
                    y_pred2[k] = 1
                else:
                    y_pred2[k] = 0
            # k = 0
            # for col in ch_sample.columns:
            #     coef[int(col)] += lr.coef_[0,k]
            #     k += 1
            coef[j,:] += lr.coef_[0,:]
            table += confusion_matrix(y_test,y_pred)
            table2 += confusion_matrix(y_test,y_pred2)
        j += 1
        table /= n_iter
        table2 /= n_iter
        coef[j-1,:] /= n_iter
        print('---------------')
        print(y_pred[:10])
        print('-')
        print(y_test[:10])
        print('-')
        print(y_prob[:10])
        print('-')
        print(lr.intercept_)
        print('-')
        print(coef)
        # print(c)
        print(table)
        print(table2)

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

    path = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\data_for_lr_1u.csv'
    pd.DataFrame.to_csv(data_out,path_or_buf=path,sep=',',index=False)

#---------------------------------------------
#transform data set to matrix format before using the data in logistic regression model
#---------------------------------------------
def TransformDataToLRmodel_NoRepetition(data_all, data_agr):
    """
    This function differs from TransformDataToLRmodel in that this generates just one user
    for each customer journey

    input
        data: input dataframe with columns:
            user_id, medium (ie. channel, line item), conversion or not of each user

    output
        data_out: dataframe with columns:
            user_id, medium (ie. channel, line item), conversion or not of each user
    """

    data_all_gruoped = pd.DataFrame.groupby(data_all, by='uid')
    data_agr_gruoped = pd.DataFrame.groupby(data_agr, by='uid')
    data_agr_user_unique = pd.Series.unique(data_agr.iloc[:,0])
    data_agr_nuser = len(data_agr_user_unique)
    data_agr_nrow = len(data_agr)
    data_all_1u = pd.DataFrame(data={'uid':data_agr.iloc[:,0], 'campaign':np.zeros(data_agr_nrow, dtype=np.int_), \
        'conversion':np.zeros(data_agr_nrow, dtype=np.int_), 'timestamp':np.zeros(data_agr_nrow, dtype=np.int_)})

    k = 0
    m = 0
    j = 0
    for name, group in data_agr_gruoped:
        npath = data_agr.iloc[k,2] + data_agr.iloc[k,3]
        nads = len(group)
        prob = data_agr.iloc[k,2] / npath
        conv = np.random.choice([0,1], p=[1-prob,prob])
        for i in group.index:
            data_all_1u.iloc[k,1] = data_agr.iloc[i,1]
            data_all_1u.iloc[k,2] = conv
            k += 1
        for _ in range(nads):
            data_all_1u.iloc[m,3] = data_all.iloc[j,3]
            m += 1
            j += 1
        j += (npath - 1) * nads

    path = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\data_all_1u.csv'
    pd.DataFrame.to_csv(data_all_1u,path_or_buf=path,sep=',',index=False)

#---------------------------------------------
#time decay model
#---------------------------------------------
def TimeDecayModel(data):
    """
    input
        data: input dataframe with four columns:
            user_id, medium (ie. channel, line item), conversion or not of each user, timestamp

    output
        medium_contribution: dataframe with two columns:
            medium, contribution
    """

    data = data.astype({'timestamp':np.float_})

    medium_unique = pd.Series.unique(data.iloc[:,1])
    n_medium = len(medium_unique)
    medium_map = {}
    medium_aux = {}
    medium_count = {}
    j = 0
    for i in medium_unique:
        medium_map[i] = j
        medium_aux[i] = 0
        medium_count[i] = 0
        j += 1

    d = {'medium':medium_unique,'contribution':np.zeros(n_medium,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    data_grouped = pd.DataFrame.groupby(data, by='uid')
    for name, group in data_grouped:
        if group.iloc[0,2] == 1:
            n_ads = group.shape[0]
            tmin = 0
            tmax = group.iloc[n_ads-1,3]
            if n_ads>1:
                for i in range(n_ads):
                    medium_aux[group.iloc[i,1]] += ( group.iloc[i,3] - tmin ) / ( tmax - tmin )
                    medium_count[group.iloc[i,1]] += 1
            else:
                medium_aux[group.iloc[0,1]] = 1
            for i in medium_aux:
                medium_loc = medium_map[i]
                medium_contribution.iloc[medium_loc,1] += medium_aux[i] / max(medium_count[i],1)
                medium_aux[i] = 0
                medium_count[i] = 0
    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])

    return(medium_contribution)

#---------------------------------------------
#time decay model
#---------------------------------------------
def SurvivalModel():
    """
    input
        data: input dataframe with four columns:
            user_id, medium (ie. channel, line item), conversion or not of each user, timestamp

    output
        medium_contribution: dataframe with two columns:
            medium, contribution
    """

    path = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\data_all_1u.csv'
    data = pd.read_csv(filepath_or_buffer=path, sep=',')
    path_tconv = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\r_dataset_tconv.csv'
    tconv = pd.read_csv(filepath_or_buffer=path_tconv, sep=',')

    beta = np.ones(12,dtype=np.float_)
    omega = np.ones(12,dtype=np.float_)
    beta_omega = np.ones(12,dtype=np.float_)
    nrep = 20
    beta_all = np.zeros((nrep,12),dtype=np.float_)
    omega_all = np.zeros((nrep,12),dtype=np.float_)

    adhazard = pd.DataFrame.copy(data)
    adhazard['p'] = np.zeros(len(adhazard),dtype=np.float_)
    adhazard['p_sum'] = np.zeros(len(adhazard),dtype=np.float_)
    adhazard['beta_den'] = np.zeros(len(adhazard),dtype=np.float_)
    adhazard['omega_den'] = np.zeros(len(adhazard),dtype=np.float_)

    data_grouped = pd.DataFrame.groupby(data, by='uid')

    for n in range(nrep):
        # count = 0
        for user, group in data_grouped:
            # count +=1
            # if count > 100:
            #     break
            index = group.index
            #if conversion p!=0
            if group.loc[index[0],'conversion'] == 1:
                suma = 0
                for i in index:
                    #which channel
                    ch = group.loc[i,'campaign']

                    #p
                    deltat = tconv.iloc[user,1] - group.loc[i,'timestamp']
                    adhazard.loc[i,'p'] = beta_omega[ch] * np.exp( - omega[ch] * deltat )
                    suma += adhazard.loc[i,'p']
                    adhazard.loc[i,'p_sum'] = suma

                    #beta denominator
                    adhazard.loc[i,'beta_den'] = 1. - np.exp( - omega[ch] * deltat )

                for i in index:
                    #which channel
                    ch = group.loc[i,'campaign']

                    #p
                    deltat = tconv.iloc[user,1] - group.loc[i,'timestamp']
                    adhazard.loc[i,'p'] /= suma

                    #omega denominator
                    adhazard.loc[i,'omega_den'] = adhazard.loc[i,'p'] * deltat + beta[ch] * deltat * np.exp( - omega[ch] * deltat )

            else:

                for i in index:
                    #which channel
                    ch = group.loc[i,'campaign']

                    #beta denominator
                    deltat = tconv.iloc[user,1] - group.loc[i,'timestamp']
                    adhazard.loc[i,'beta_den'] = 1. - np.exp( - omega[ch] * deltat )

                    #omega denominator
                    adhazard.loc[i,'omega_den'] = beta[ch] * deltat * np.exp( - omega[ch] * deltat )

        #update omega and beta
        data_grouped_ch = pd.DataFrame.groupby(adhazard, by='campaign')
        for ch, ch_group in data_grouped_ch:
            beta_den = ch_group['beta_den'].sum()
            if beta_den > 1e-6:
                beta[ch] = ch_group['p'].sum() / beta_den
            beta_all[n,ch] = beta[ch]
            omega_den = ch_group['omega_den'].sum()
            if omega_den > 1e-6:
                omega[ch] = ch_group['p'].sum() / omega_den
            omega_all[n,ch] = omega[ch]
            beta_omega[ch] = beta[ch] * omega[ch]

        print(n)

    betadf = pd.DataFrame(data=beta_all)
    omegadf = pd.DataFrame(data=omega_all)

    path_beta = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\ad_hazard_beta.csv'
    pd.DataFrame.to_csv(betadf,path_or_buf=path_beta,sep=',',index=False)
    path_omega = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\ad_hazard_omega.csv'
    pd.DataFrame.to_csv(omegadf,path_or_buf=path_omega,sep=',',index=False)

def SurvivalModelCont():
    """
    input
        data: input dataframe with four columns:
            user_id, medium (ie. channel, line item), conversion or not of each user, timestamp

    output
        medium_contribution: dataframe with two columns:
            medium, contribution
    """

    path = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\data_all_1u.csv'
    data = pd.read_csv(filepath_or_buffer=path, sep=',')
    path_tconv = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\r_dataset_tconv.csv'
    tconv = pd.read_csv(filepath_or_buffer=path_tconv, sep=',')

    medium_unique = pd.Series.unique(data.iloc[:,1])
    n_medium = len(medium_unique)
    medium_map = {}
    medium_aux = {}
    medium_count = {}
    j = 0
    for i in medium_unique:
        medium_map[i] = j
        medium_aux[i] = 0
        medium_count[i] = 0
        j += 1

    d = {'medium':medium_unique,'contribution':np.zeros(n_medium,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)

    path_beta = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\ad_hazard_beta.csv'
    betadf = pd.read_csv(filepath_or_buffer=path_beta, sep=',')
    path_omega = r'C:\Users\sesig\Documents\master data science\tfm\r_dataset_cleaned\ad_hazard_omega.csv'
    omegadf = pd.read_csv(filepath_or_buffer=path_omega, sep=',')

    beta = betadf.iloc[-1,:]
    omega = omegadf.iloc[-1,:]

    data_grouped = pd.DataFrame.groupby(data, by='uid')
    for user, group in data_grouped:
        index = group.index
        if group.loc[index[0],'conversion'] == 1:
            suma = 0
            p = np.zeros(len(index),dtype=np.float_)
            j = 0
            for i in index:
                #which channel
                ch = group.loc[i,'campaign']

                #p
                deltat = tconv.iloc[user,1] - group.loc[i,'timestamp']
                p[j] = beta[ch] * omega[ch] * np.exp( - omega[ch] * deltat )
                suma += p[j]
                j += 1
            j = 0
            for i in index:
                #chich channel
                ch = group.loc[i,'campaign']

                p[j] /= suma

                medium_contribution.iloc[ch,1] += p[j]

                j += 1

    medium_contribution.iloc[:,1] /= np.sum(medium_contribution.iloc[:,1])

    return(medium_contribution)

#---------------------------------------------
#plot contribution of the channel for each model
#---------------------------------------------
def contribution(x,select):
    """
    input
        x: dataframe
            each row is a channel and each column its contribution to the model
    """

    n_ch = 3
    x_loc = np.arange(n_ch)
    width = 0.1

    models1 = ['last', 'linear', 'time_decay']
    models2 = ['prob', 'lr', 'ad_hazard']

    labels1 = ['Last', 'Linear (prop)', 'Time Decay']
    labels2 = ['Probabilistic', 'L. Regression', 'Additive Hazard']

    #channels 1 to 6
    if select == 0:

        # ch0_1 = x.iloc[0,2:5]
        # ch0_2 = x.iloc[0,8:]
        ch0_1 = x.loc[0,models1]
        ch0_2 = x.loc[0,models2]

        # ch1_1 = x.iloc[1,2:5]
        # ch1_2 = x.iloc[1,8:]
        ch1_1 = x.loc[1,models1]
        ch1_2 = x.loc[1,models2]


        # ch2_1 = x.iloc[2,2:5]
        # ch2_2 = x.iloc[2,8:]
        ch2_1 = x.loc[2,models1]
        ch2_2 = x.loc[2,models2]

        # ch3_1 = x.iloc[3,2:5]
        # ch3_2 = x.iloc[3,8:]
        ch3_1 = x.loc[3,models1]
        ch3_2 = x.loc[3,models2]

        # ch4_1 = x.iloc[4,2:5]
        # ch4_2 = x.iloc[4,8:]
        ch4_1 = x.loc[4,models1]
        ch4_2 = x.loc[4,models2]

        # ch5_1 = x.iloc[5,2:5]
        # ch5_2 = x.iloc[5,8:]
        ch5_1 = x.loc[5,models1]
        ch5_2 = x.loc[5,models2]

    #channels 7 to 12
    elif select == 1:
        # ch0_1 = x.iloc[6,2:5]
        # ch0_2 = x.iloc[6,7:]
        ch0_1 = x.loc[0,models1]
        ch0_2 = x.loc[0,models2]

        # ch1_1 = x.iloc[7,2:5]
        # ch1_2 = x.iloc[7,7:]
        ch1_1 = x.loc[1,models1]
        ch1_2 = x.loc[1,models2]

        # ch2_1 = x.iloc[8,2:5]
        # ch2_2 = x.iloc[8,7:]
        ch2_1 = x.loc[2,models1]
        ch2_2 = x.loc[2,models2]

        # ch3_1 = x.iloc[9,2:5]
        # ch3_2 = x.iloc[9,7:]
        ch3_1 = x.loc[3,models1]
        ch3_2 = x.loc[3,models2]

        # ch4_1 = x.iloc[10,2:5]
        # ch4_2 = x.iloc[10,7:]
        ch4_1 = x.loc[4,models1]
        ch4_2 = x.loc[4,models2]

        # ch5_1 = x.iloc[11,2:5]
        # ch5_2 = x.iloc[11,7:]
        ch5_1 = x.loc[5,models1]
        ch5_2 = x.loc[5,models2]
    else:
        return('invalid number. 0 for first 6 channels. 1 for rest of channels')

    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(5.9055,3))

    ax[0].bar(x_loc - (width / 2 + 2 * width), ch0_1, width=width, label='0')
    ax[0].bar(x_loc - (width / 2 + 1 * width), ch1_1, width=width, label='1')
    ax[0].bar(x_loc - (width / 2 + 0 * width), ch2_1, width=width, label='2')
    ax[0].bar(x_loc + (width / 2 + 0 * width), ch3_1, width=width, label='3')
    ax[0].bar(x_loc + (width / 2 + 1 * width), ch4_1, width=width, label='4')
    ax[0].bar(x_loc + (width / 2 + 2 * width), ch5_1, width=width, label='5')

    ax[1].bar(x_loc - (width / 2 + 2 * width), ch0_2, width=width, label='0')
    ax[1].bar(x_loc - (width / 2 + 1 * width), ch1_2, width=width, label='1')
    ax[1].bar(x_loc - (width / 2 + 0 * width), ch2_2, width=width, label='2')
    ax[1].bar(x_loc + (width / 2 + 0 * width), ch3_2, width=width, label='3')
    ax[1].bar(x_loc + (width / 2 + 1 * width), ch4_2, width=width, label='4')
    ax[1].bar(x_loc + (width / 2 + 2 * width), ch5_2, width=width, label='5')

    ax[0].set_xticks(x_loc)
    ax[1].set_xticks(x_loc)

    ax[0].set_xticklabels(labels1)
    ax[1].set_xticklabels(labels2)

    # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5,1.3),fancybox=False, shadow=False, ncol=6)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5,1.4),fancybox=False, shadow=False, ncol=6)
    # ax[1].legend()
    # fig.legend()
    plt.show()

    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'text.usetex': True
    })

    # plt.savefig(r'C:\Users\sesig\Documents\master data science\tfm\project\imagenes\datadriven_ch_cont.pgf')

#---------------------------------------------
#plot contribution of the channel for each model
#---------------------------------------------
def contribution2(x,select):
    """
    input
        x: dataframe
            each row is a channel and each column its contribution to the model
    """

    n_ch = 6
    x_loc = np.arange(n_ch)
    width = 0.1

    models1 = ['last', 'linear', 'first']
    models2 = ['bathtub', 'linear_same', 'pos_decay']

    labels1 = ['Ch 0', 'Ch 1', 'Ch 2', 'Ch 3', 'Ch 4', 'Ch 5', 'Ch 6']

    #channels 1 to 6
    if select == 0:

        ch0_1 = x.loc[0:5,'last']
        label0 = 'Last'

        ch1_1 = x.loc[0:5,'linear']
        label1 = 'Linear (prop)'

        # ch2_1 = x.loc[0:5,'first']
        # label2 = 'First'
        ch2_1 = x.loc[0:5,'time_decay']
        label2 = 'Time Decay'

        # ch3_1 = x.loc[0:5,'bathtub']
        # label3 = 'Bathtub'
        ch3_1 = x.loc[0:5,'prob']
        label3 = 'Probabilistic'

        # ch4_1 = x.loc[0:5,'pos_decay']
        # label4 = 'Pos. Decay'
        ch4_1 = x.loc[0:5,'lr3']
        label4 = 'L. Regression'

        # ch5_1 = x.loc[0:5,'linear_same']
        # label5 = 'Linear (same)'
        ch5_1 = x.loc[0:5,'ad_hazard']
        label5 = 'Ad. Hazard'

    #channels 7 to 12
    elif select == 1:
        # ch0_1 = x.iloc[6,2:5]
        # ch0_2 = x.iloc[6,7:]
        ch0_1 = x.loc[0,models1]
        ch0_2 = x.loc[0,models2]

        # ch1_1 = x.iloc[7,2:5]
        # ch1_2 = x.iloc[7,7:]
        ch1_1 = x.loc[1,models1]
        ch1_2 = x.loc[1,models2]

        # ch2_1 = x.iloc[8,2:5]
        # ch2_2 = x.iloc[8,7:]
        ch2_1 = x.loc[2,models1]
        ch2_2 = x.loc[2,models2]

        # ch3_1 = x.iloc[9,2:5]
        # ch3_2 = x.iloc[9,7:]
        ch3_1 = x.loc[3,models1]
        ch3_2 = x.loc[3,models2]

        # ch4_1 = x.iloc[10,2:5]
        # ch4_2 = x.iloc[10,7:]
        ch4_1 = x.loc[4,models1]
        ch4_2 = x.loc[4,models2]

        # ch5_1 = x.iloc[11,2:5]
        # ch5_2 = x.iloc[11,7:]
        ch5_1 = x.loc[5,models1]
        ch5_2 = x.loc[5,models2]
    else:
        return('invalid number. 0 for first 6 channels. 1 for rest of channels')

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5.9055,3))
    
    ax.bar(x_loc - (width / 2 + 2 * width), ch0_1, width=width, label=label0)
    ax.bar(x_loc - (width / 2 + 1 * width), ch1_1, width=width, label=label1)
    ax.bar(x_loc - (width / 2 + 0 * width), ch2_1, width=width, label=label2)
    ax.bar(x_loc + (width / 2 + 0 * width), ch3_1, width=width, label=label3)
    ax.bar(x_loc + (width / 2 + 1 * width), ch4_1, width=width, label=label4)
    ax.bar(x_loc + (width / 2 + 2 * width), ch5_1, width=width, label=label5)

    ax.set_xticks(x_loc)

    ax.set_ylim([0,0.5])

    # ax.set_yscale('log')

    ax.set_xticklabels(labels1)

    ax.legend(loc='best',fancybox=False, shadow=False, ncol=1)

    # plt.show()

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'text.usetex': True
    })

    plt.savefig(r'C:\Users\sesig\Documents\master data science\tfm\project\imagenes\datadriven_ch_cont.pgf')
    # plt.savefig(r'C:\Users\sesig\Documents\master data science\tfm\project\imagenes\simple_ch_cont.pgf')

#---------------------------------------------
#only runs the code if executed as main
#---------------------------------------------
if __name__== '__main__':
    print('Running this file as the main file does nothing')