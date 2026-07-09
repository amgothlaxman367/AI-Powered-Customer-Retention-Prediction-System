import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from log_code import setup_logging
logger = setup_logging("missing_data_handling")


def missing(X_train,X_test):

    try:

        logger.info(f"Before Missing values Train data {X_train.isnull().sum()}")
        logger.info(f"Before Missing values Test data {X_test.isnull().sum()}")

        for i in X_train.columns:
            if X_train[i].isnull().sum() > 0:
                X_train[i+"_tc"]=X_train[i].copy()
                X_test[i+"_tc"]=X_test[i].copy()
                t1=X_train[i].dropna().sample(X_train[i].isnull().sum(),random_state=42)
                t2=X_test[i].dropna().sample(X_test[i].isnull().sum(),random_state=42)
                t1.index=X_train[X_train[i].isnull()].index
                t2.index=X_test[X_test[i].isnull()].index
                X_train.loc[X_train[i].isnull(),i+"_tc"]=t1
                X_test.loc[X_test[i].isnull(),i+"_tc"]=t2
                X_train= X_train.drop([i],axis=1)
                X_test= X_test.drop([i],axis=1)
        logger.info(f"After Missing values Train data {X_train.isnull().sum()}")
        logger.info(f"After Missing values Test data {X_test.isnull().sum()}")
        return X_train, X_test


    except Exception as e:
       er_ty, er_msg, er_line = sys.exc_info()
       logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")






