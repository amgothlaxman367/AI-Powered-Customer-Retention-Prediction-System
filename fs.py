import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np

from scipy.constants import value

from log_code import setup_logging
logger = setup_logging("fs")
from sklearn.feature_selection import VarianceThreshold
constant_reg=VarianceThreshold(threshold=0.0)
quasi_reg=VarianceThreshold(threshold=0.1)
from scipy.stats import pearsonr

def best_columns(X_train_num , X_test_num,y_train):
    try:

        logger.info(f" Before Train Data : {X_train_num.shape} : {X_train_num.columns}")
        logger.info(f" Before Test Data : {X_test_num.shape} : {X_test_num.columns}")

        constant_reg.fit(X_train_num)
        logger.info(f"Number of Columns are good {sum(constant_reg.get_support())}")
        logger.info(f"Number of Columns are Bad : {sum(~constant_reg.get_support())}")
        logger.info(f"Columns to remove : {X_train_num.columns[~constant_reg.get_support()]}")

        X_train_num = X_train_num.drop(['SeniorCitizen_yeo_trim'], axis=1)
        X_test_num = X_test_num.drop(['SeniorCitizen_yeo_trim'], axis=1)

        # using constant technique we remove 2 columns
        # so we have 4 know
        # out of 4 using quasi constant we can remove more let's see
        logger.info(f"==========================================================")
        quasi_reg.fit(X_train_num)
        logger.info(f"Number of Columns are good {sum(quasi_reg.get_support())}")
        logger.info(f"Number of Columns are Bad : {sum(~quasi_reg.get_support())}")
        logger.info(f"Columns to remove : {X_train_num.columns[~quasi_reg.get_support()]}")

       # logger.info(f"{X_train_num.columns}")
       # values=[]
       #  for i in X_train_num.columns:
       #     values.append(pearsonr(X_train_num[i],y_train))
       # values = np.array(values)
       # p_values=values[ : , 1]
       # s=pd.Series(p_values,index=X_train_num.columns)
       # plt.figure(figsize = (5,3))
       # s.plot.bar()
       # plt.show()

        X_train_num = X_train_num.drop(['MonthlyCharges_yeo_trim'], axis=1)
        X_test_num = X_test_num.drop(['MonthlyCharges_yeo_trim'], axis=1)
        return X_train_num, X_test_num

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

