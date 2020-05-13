#####################################################
#logistic regression model
#
#this module implements the logistic regession model for
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

    n_iter = 10
    test_size = 0.25
    p_obs = 0.3
    p_var = 0.4

    ch = np.arange(0,11,1,dtype=np.int_)
    d = {'medium':ch, 'contribution':np.zeros(12,dtype=np.float_)}
    medium_contribution = pd.DataFrame(data=d)
    

    for i in range(n_iter):
        data_sample = data.sample(frac=p_obs, random_state=i, axis=0)
        ch_sample = data_sample[[0,1,2,3,4,5,6,7,8,9,10,11]].sample(frac=p_var, random_state=i, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(ch_sample.values, data_sample['conversion'], test_size=test_size, random_state=i, stratify=data_sample['conversion'])
        lr = LogisticRegression(penalty='l2', C=1., fit_intercept=True, solver='lbfgs')
        lr.fit(x_train, y_train)
        lr_pred = lr.predict(x_test)
        print(lr.get_params())




#---------------------------------------------
#only runs the code if executed as main
#---------------------------------------------
if __name__== '__main__':
    print('Running this file as the main file does nothing')