#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Accidents Prediction App
This app predicts the most accident prone state of USA
""")
st.write('---')

df=pd.read_csv('Train.csv')

X_train=df[['Item_Weight','Item_Visibility','Item_MRP',]]
y_train=df['Item_Outlet_Sales']

st.sidebar.header('User Input Features')

def user_input_features():
    drvr_fatl_col_bmiles=st.sidebar.slider('No. of drivers involved in fatal collision',int(X_train.Item_Weight.min()),int(X_train.Item_Weight.max()),int(X_train.Item_Weight.mean()))
    perc_fatl_speed=st.sidebar.slider('% of drivers involved in over-speeding',float(X_train.Item_Visibility.min()),float(X_train.Item_Visibility.max()),float(X_train.Item_Visibility.mean()))
    perc_fatl_alcohol=st.sidebar.slider('% of drivers alcohol impaired',int(X_train.Item_MRP.min()),int(X_train.Item_MRP.max()),int(X_train.Item_MRP.mean()))

    data={'drvr_fatl_col_bmiles':drvr_fatl_col_bmiles,
    'perc_fatl_speed':perc_fatl_speed,
    'perc_fatl_alcohol':perc_fatl_alcohol,
    }

    features=pd.DataFrame(data,index=[0])
    return features

df1=user_input_features()
st.write(df1)
st.write('---')


clf=RandomForestClassifier()
clf.fit(X_train,y_train)

prediction=clf.predict(df1)

st.write("""
### The most probable state
""")
st.write(prediction)

