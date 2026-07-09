import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from log_code import setup_logging
logger = setup_logging("vt_outlier")
from  scipy.stats import yeojohnson



def varibale_outlier_handling(X_train_num,X_test_num):

    try:

       logger.info(f" Train Data : {X_train_num.shape} : \n :{X_train_num.columns}" )
       logger.info(f" Test Data  : {X_test_num.shape} : \n :{X_test_num.columns}")

       '''
       apply yeojohnson for each column and save in new column then delete old column on
       top of it apply outliers and remove old column
       '''
       for k in X_train_num.columns:
           X_train_num[k+"_yeo"],lam_value = yeojohnson(X_train_num[k])
           X_test_num[k+"_yeo"],lam_value = yeojohnson(X_test_num[k])

           X_train_num=X_train_num.drop([k],axis=1)
           X_test_num=X_test_num.drop([k],axis=1)

           iqr= X_train_num[k+"_yeo"].quantile(0.75) - X_train_num[k+"_yeo"].quantile(0.25)
           lower_limit = X_train_num[k+"_yeo"].quantile(0.25) - (1.5*iqr)
           upper_limit = X_train_num[k+"_yeo"].quantile(0.75) + (1.5*iqr)

           X_train_num[k+"_yeo_trim"] = np.where(X_train_num[k+"_yeo"] < lower_limit , lower_limit ,
           np.where(X_train_num[k+"_yeo"] > upper_limit , upper_limit , X_train_num[k+"_yeo"]))

           X_test_num[k + "_yeo_trim"] = np.where(X_test_num[k + "_yeo"] < lower_limit, lower_limit,
           np.where(X_test_num[k + "_yeo"] > upper_limit, upper_limit,X_test_num[k + "_yeo"]))

           X_train_num = X_train_num.drop([k+"_yeo"],axis=1)
           X_test_num = X_test_num.drop([k+"_yeo"],axis=1)

       ''' plt.figure(figsize=(5, 3))
       for i in X_train_num.columns:
           plt.title(i)
           # X_train_num[i].plot(kind = 'kde')
           sns.displot(X_train_num[i], kind='kde')
           plt.show()

       plt.figure(figsize=(5, 3))
       for i in X_train_num.columns:
           plt.title(i)
           sns.boxplot(x = X_train_num[i])
           plt.show()
       '''

       logger.info(f" After Train Data : {X_train_num.shape} : \n :{X_train_num.columns}")
       logger.info(f" After Test Data  : {X_test_num.shape} : \n :{X_test_num.columns}")

       return X_train_num,X_test_num


    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")