#%%
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
import warnings
import time
import os
import sys


DeprecationWarning('ignore')
warnings.filterwarnings('ignore',message="don't have warning")

#%%
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 


#%%
df=pd.read_csv('housing.csv')

#%%
df.head()

#%%
df.describe()

#%%
df.isnull().sum()

#%%
import seaborn as sns
sns.distplot(df.total_bedrooms.dropna())
#%%
train, test = train_test_split(df,test_size=0.2,random_state = 12)


#%%
def bed(df):
    mean=537.87055
    df['total_bedrooms'].fillna(mean,inplace=True)
    return df

def label_encode(df):
    from sklearn.preprocessing import LabelEncoder 
    label = LabelEncoder()
    df['ocean_proximity'] =  label.fit_transform(df['ocean_proximity'])
    return df

def encode_feature(df):
    df=bed(df)
    df = label_encode(df)
    return(df)
train = encode_feature(train)
test = encode_feature(test)


#%%
def x_and_y(df):
    x = df.drop(["median_house_value","ocean_proximity"],axis=1)
    y = df["median_house_value"]
    return x,y
x_train,y_train = x_and_y(train)
x_test,y_test = x_and_y(test)


#%%
from sklearn.metrics import r2_score
reg = LinearRegression()
reg.fit(x_train,y_train)
pridict=reg.predict(x_test)
print("LinearRegression\n",r2_score(y_test,pridict)*100)

#%%

reg = LinearRegression()
reg.fit(x_train,y_train)
pridict=reg.predict(x_train)
print("LinearRegression\n",r2_score(y_train,pridict)*100)

#%%
