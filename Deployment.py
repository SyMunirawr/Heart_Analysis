# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:18:10 2022

@author: Tuf
"""

import pickle
import os
import numpy as np
from PIL import Image
import streamlit as st

#%% TRAINED MODEL LOADING
MODEL_PATH = os.path.join(os.getcwd(),'model','best_model.pkl')
with open(MODEL_PATH,'rb') as file:
        model = pickle.load(file)
        
X=['cp','thall', 'age', 'trtbps', 'chol', 'thalachh','caa']

#%% To test if the predicted outcome is correct or wrong
# #input example, info from 0 rows in dataframe of selected features
# X_new=[2,2,37,130,250,187,0]
# outcome=model.predict(np.expand_dims(np.array(X_new),axis=0))#
# print(outcome)

#%% deployment : model.predict#%% To create a form

image = Image.open('heart image.jpg')
st.image(image)

st.header('Diagnosis of heart disease (Angiographic Disease status)')

with st.form("Patient's info"):
    st.write("This app is to predict the status of angiographic disease.")
    age = st.slider('How old are you?', 20, 100)
    st.write("I'm ", age, 'years old')
    #Chest pain type,cp
    cp_display = ("Asymptomatic","Typical Angina","Atypical Angina",
                  "Non-anginal Pain")
    options = list(range(len(cp_display)))
    cp = st.selectbox("Chest Pain Type", options, 
                      format_func=lambda x:cp_display[x])
    st.write(cp)
    trtbps = int(st.number_input('Please key in your resting blood pressure'))
    #Thalium Stress Test result,thall
    thall_display=('Fixed Defect','Normal','Reversable Defect')
    options_ = list(range(len(thall_display)))
    thall = st.selectbox("Thalium Stress Test result", 
                         options_, format_func=lambda x: thall_display[x])
    st.write(thall)
    chol=int(st.number_input('Please key in your cholesterol reading'))
    thalachh=int(st.number_input('Please key in your maximum heart rate achieved'))
    caa = st.slider('Please select your number of major vessels', 0,3)
    st.write("Number of major vessels : ", caa)
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Age", age, "Chest pain type", cp," The resting blood pressure",
                 trtbps,"Thalium Stress Test result", thall,"Cholesterol", chol,
                 "maximum heart rate achieved", thalachh, 
                 "Number of major vessels", caa)
        X_new=[cp,thall,age,trtbps,chol,thalachh,caa]
        outcome = model.predict(np.expand_dims([age,cp,trtbps,thall,chol,thalachh,caa]
                                ,axis=0))
        outcome_dict = {0:'Less than 50% chance the diameter of coronary artery is narrowing', 
                        1:'More than 50% chance the diameter of coronary artery is narrowing'}
        st.write(outcome_dict[outcome[0]])
        
        if outcome == 1:
             st.warning('Oh no! You have a higher risk of getting heart disease!')
        else:
            st.balloons()
            st.success('You are healthy and have a less chance of getting heart disease!')