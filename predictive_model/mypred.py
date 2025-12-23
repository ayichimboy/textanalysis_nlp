# Import libraries and write functions 

import pandas as pd               
import numpy as np                  
import matplotlib.pyplot as plt                   
import seaborn as sns           
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_data(input_df):
    scaler = StandardScaler()
    X = input_df.iloc[:,0]
    y = input_df.iloc[:1]
    
    X_train, X_test, y_train, y_test=train_test_split(X, y, 
                                                       test_size=0.2, 
                                                       random_state=42)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__" :
    print("completed code")
    