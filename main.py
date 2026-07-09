
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import os
import random
import warnings
warnings.filterwarnings("ignore")
import sys
from log_code import setup_logging
logger=setup_logging("main")
from sklearn.model_selection import train_test_split
from missing_data_handling import missing
from vt_outlier import varibale_outlier_handling
from fs import best_columns
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler # Z_score
#from all_models import common
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle

class CHURN:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(path)
            logger.info(f"The shape of the data was : {self.df.shape}")
            logger.info(f"Before Number of Null values : {self.df.isnull().sum()}")
            self.df=self.df.drop(['customerID'],axis=1)
            logger.info(f"After Number of Null values : {self.df.shape}")
            logger.info(f"After Number of Null values : {self.df.isnull().sum()}")
            logger.info(f"Data Set Informantion {self.df.info()}")
            logger.info(f"=========Before Updated dataset Size ============")
            logger.info({self.df.shape})

            # Adding a new column "SIM"

            def add_sim(row):  # row is a parameter that receives one row of the DataFrame. from the apply() function.
                sim_providers = ["Jio", "Airtel", "Vi", "BSNL"]
                return  random.choice(sim_providers)
            self.df["Sim"]=self.df.apply(add_sim,axis=1)    # apply() applies a function to each row or column of a DataFrame.

            logger.info(f'After updated file is {self.df}')
            logger.info(f'After updated dataset Size is: {self.df.shape}')
            logger.info(f'After updated dataset colums is: {self.df.columns}')
            logger.info(self.df.info())
            logger.info(f"After Number of Null values : {self.df.isnull().sum()}")
            logger.info(self.df.dtypes)
            logger.info('=====================convert string into float========================')
            logger.info("=================Before=========================")
            logger.info(f"{self.df['TotalCharges'].dtype}")
            self.df["TotalCharges"]=self.df["TotalCharges"].replace(" ", np.nan)
            self.df["TotalCharges"]=pd.to_numeric(self.df["TotalCharges"])
          #  logger.info(f"After convert string into float is {self.df['TotalCharges'].dtype}")
            logger.info("=================After=========================")
            logger.info(f'{self.df.dtypes}')
            logger.info(f"{self.df.isnull().sum()}")

            # splitting the data into X and Y
            self.X = self.df.drop("Churn", axis=1)
            self.y = self.df["Churn"]

           # self.X=self.df.iloc[:,:-1] # independent
           # self.y=self.df.iloc[:,-2] # dependent
            self.y = self.y.map({'Yes': 1, 'No': 0}).astype(int)
            logger.info(f"Independent data : {self.X.shape} : \n : {self.X.isnull().sum()}")
            logger.info(f"Dependent data : {self.y.shape} : \n : {self.y.isnull().sum()}")
            # logger.info(self.y.unique())
            logger.info(f'{self.y.value_counts()}')

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            logger.info(f"Train Dataset size : {self.X_train.shape} => {self.y_train.shape}")
            logger.info(f"Test Dataset size : {self.X_test.shape} => {self.y_test.shape}")
        except Exception as e:
          er_ty, er_msg, er_line = sys.exc_info()
          logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def handling_missing_values(self):

        try:

            '''
            here we are going to implement to random sample imputation technique
            '''

            logger.info(f"Before Missing values Train data {self.X_train.isnull().sum()}")
            logger.info(f"Before Missing values Test data {self.X_test.isnull().sum()}")

            self.X_train , self.X_test= missing(self.X_train, self.X_test)

            logger.info(f"After Missing values Train data {self.X_train.isnull().sum()}")
            logger.info(f"After Missing values Test data {self.X_test.isnull().sum()}")

            '''
            since missing values completed we need to divide the data into 2 parts numerical column and categorical column
            '''

            logger.info(f"Total data in X_train {self.X_train.shape}")
            logger.info(f"Total data in X_test {self.X_test.shape}")

            self.X_train_numerical=self.X_train.select_dtypes(exclude='str')
            self.X_train_categorical=self.X_train.select_dtypes(include='str')
            self.X_test_numerical=self.X_test.select_dtypes(exclude='str')
            self.X_test_categorical=self.X_test.select_dtypes(include='str')

            logger.info(f"===================X_train_saperation_detaile=================")
            logger.info(f" X_train_data {self.X_train.shape}")
            logger.info(f"X_train_numerical {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f"X_train_categorical {self.X_train_categorical.shape} : \n : {self.X_train_categorical.columns}")
            logger.info(f"===================X_test_saperation_detaile=================")
            logger.info(f" X_test_data {self.X_test.shape}")
            logger.info(f"X_test_numerical {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            logger.info(f"X_test_categorical {self.X_test_categorical.shape} : \n : {self.X_test_categorical.columns}")

            '''
            from the above data we got to know that we training total we have 20 columns in 20 we have 4 num 16 categorical
            this info this for training data(X_train) and also for testing data (X_test)
            '''
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def vt_outlier_handling(self):
        try:
            logger.info(f" Before X_train_numerical : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f" Before X_test_numerical : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            self.X_train_numerical, self.X_test_numerical = varibale_outlier_handling(self.X_train_numerical,self.X_test_numerical)
            logger.info(f" After X_train_numerical : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f" After X_test_numerical : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")



    def feature_selection(self):
        try:
            logger.info(f"Before feature Selection : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f"Before feature Selection : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            self.X_train_numerical, self.X_test_numerical = best_columns(self.X_train_numerical, self.X_test_numerical, self.y_train)
            logger.info(f"After feature Selection  : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(f"After feature Selection  : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")


    def cat_to_num(self):
        try:

           logger.info(f"=======Categorical Columns======== : {self.X_train_categorical.columns}")

           one_hot_obj = OneHotEncoder(drop='first')
           one_hot_obj.fit_transform(self.X_train_categorical[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
          'PaperlessBilling', 'PaymentMethod', 'Sim']])
           values_1 = one_hot_obj.transform(self.X_train_categorical[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
         'PaperlessBilling', 'PaymentMethod', 'Sim']]).toarray()
           values_2 = one_hot_obj.transform(self.X_test_categorical[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
         'PaperlessBilling', 'PaymentMethod', 'Sim']]).toarray()
           t1 = pd.DataFrame(data=values_1, columns=one_hot_obj.get_feature_names_out())
           t2 = pd.DataFrame(data=values_2, columns=one_hot_obj.get_feature_names_out())
           self.X_train_categorical.reset_index(drop=True, inplace=True)
           self.X_test_categorical.reset_index(drop=True, inplace=True)
           t1.reset_index(drop=True, inplace=True)
           t2.reset_index(drop=True, inplace=True)
           self.X_train_categorical_new = pd.concat([self.X_train_categorical, t1], axis=1)
           self.X_test_categorical_new = pd.concat([self.X_test_categorical, t2], axis=1)
           self.X_train_categorical_new = self.X_train_categorical_new.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
         'PaperlessBilling', 'PaymentMethod', 'Sim'], axis=1)
           self.X_test_categorical_new = self.X_test_categorical_new.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
         'PaperlessBilling', 'PaymentMethod', 'Sim'], axis=1)

           logger.info(f"After Converting categorical data to Numerical")
           logger.info( f"X_train_cat_data :{self.X_train_categorical_new.shape} : \n : {self.X_train_categorical_new.columns} : {self.X_train_categorical_new.isnull().sum()}")
           logger.info(f"X_test_cat_data : {self.X_test_categorical_new.shape} : \n : {self.X_test_categorical_new.columns} : {self.X_test_categorical_new.isnull().sum()}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")


    def balance_data(self):
        try:

            logger.info(f"========================Before Checking Data is Balanced or Not============================")
            logger.info(f"Number of rows for Good Customers : {1} -> : {sum(self.y_train == 1) }")
            logger.info(f"Number of rows for Bad Customers : {0} -> : {sum(self.y_train == 0)}")
            smote_reg=SMOTE(random_state=42)
            self.X_train_categorical_new,self.y_train_up = smote_reg.fit_resample(self.X_train_categorical_new,self.y_train)
            logger.info(f"===========After Checking Data is Balanced or Not=============")
            logger.info(f"Number of rows for Good Customers : {1} -> : {sum(self.y_train_up == 1)}")
            logger.info(f"Number of rows for Bad Customers : {0} -> : {sum(self.y_train_up == 0)}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def scaling_data(self):
        try:

            print(self.X_train_categorical_new)
            sc = StandardScaler()
            sc.fit(self.X_train_categorical_new)
            self.final_training_data_up_scaled = sc.transform(self.X_train_categorical_new)
            self.final_testing_data_scaled = sc.transform(self.X_test_categorical_new)
            print(self.final_training_data_up_scaled)
            # common(self.final_training_data_up_scaled,self.y_train_up,self.final_testing_data_scaled,self.y_test)
            logger.info(f"===========Training Logistic Regression==============")
            self.reg = LogisticRegression()
            self.reg.fit(self.final_training_data_up_scaled, self.y_train_up)  # trainnig completed
            self.y_test_prediction = self.reg.predict(self.final_testing_data_scaled)
            logger.info(f"Test Data Accuracy :{accuracy_score(self.y_test, self.y_test_prediction)}")
            logger.info(f"Confusion Matrix : {confusion_matrix(self.y_test, self.y_test_prediction)}")
            logger.info(f"Classification Report : {classification_report(self.y_test, self.y_test_prediction)}")
            logger.info(
                f"==============Saving the Scaled and Logistic Regression Model into Pickle File=================")
            logger.info(f' ==========================={self.X_train_categorical_new.columns} : \n : {self.X_train_categorical_new.shape}=========================')
            logger.info(f' ==========================={self.X_test_categorical_new.columns} : \n : {self.X_test_categorical_new.shape}=========================')
            with open("standard_scaler.pkl", "wb") as f:
                pickle.dump(sc, f)

            with open("WA_Fn-UseC_-Telco-Customer-Churn.pkl", "wb") as f1:
                pickle.dump(self.reg, f1)




        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")



if __name__ == '__main__':
   try:

     obj=CHURN("WA_Fn-UseC_-Telco-Customer-Churn.csv")
     obj.handling_missing_values()
     obj.vt_outlier_handling()
     obj.feature_selection()
     obj.cat_to_num()
     obj.balance_data()
     obj.scaling_data()

   except Exception as e:
     er_ty, er_msg, er_line = sys.exc_info()
     logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")







