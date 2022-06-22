# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:21:37 2022

@author: Saiyidah Munirah
"""

import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression    
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix

#%% Statics
CSV_PATH = os.path.join(os.getcwd(),'Dataset','heart.csv')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'model','best_model.pkl')
#%% Function
def plot_con(df,con_column):
    for con in con_column:
        plt.figure()
        sns.distplot(df[con])
        plt.show()
        
def plot_cat(df,cat_column):
    for cat in cat_column:
        plt.figure()
        sns.countplot(df[cat],hue=df['output'])#hue=categorical target
        plt.show()        
        
def cramers_corrected_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n    
    r,k = confusion_matrix.shape    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
#%% EDA
# Description of variables
#age - age in years

#sex:sex (1:male; 0:female)
#cp:chest pain type
#(1:typical angina;2:atypical angina;3:non-anginal pain;0:asymptomatic)
#trstbps - resting blood pressure (in mm Hg on admission to the hospital)
#chol - cholestoral in mg/dl
#fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
#restecg:resting electrocardiographic results 
#(1:normal; 2:having ST-T wave abnormality; 0:hypertrophy)
#thalach:maximum heart rate achieved
#exng:exercise induced angina (1 = yes; 0 = no)
#oldpeak:ST depression induced by exercise relative to rest
#slp:slope of the peak exercise ST segment(2:upsloping; 1:flat;0:downsloping)
#caa:number of major vessels (0-3) colored by flourosopy
#thal:Thalium Stress Test result (2:normal;1:fixed defect;3: reversable defect)
#output: diagnosis of heart disease (angiographic disease status)
# #0: < 50% diameter narrowing. less chance of heart disease
# #1: > 50% diameter narrowing. more chance of heart disease

#Step 1: Data Loading
df=pd.read_csv(CSV_PATH)

#%%
#Step 2: Data Inspection
#1) variables' type, distribution and descriptive statistics
df.info()
stats=df.describe().T
df.boxplot()

#2) check for outliers, NaNs and duplicates
df.duplicated().sum()
df.isna().sum() 
msno.bar(df)

#Based on the initial inspection:
# The data consists of 1 duplicate and no NaNs

# To impute the null in 'thall' column
df['thall']=df['thall'].replace(0, np.nan)
df['thall'] = df['thall'].fillna(df['thall'].mode()[0])
df.info()

#3) Visualize the distribution of dataset
cat_column=['sex','cp','fbs','restecg','exng','slp','thall','output']
con_column=['age','trtbps','chol','thalachh','caa']
        
plot_con(df,con_column)       
plot_cat(df,cat_column)    

# Based on the plots:
# Female and male have relatively equal chances of getting heart disease based
# on relatively a small difference. 
# Patients with chest pain type with asymptomatic,dont have exercise induced angina, 
# Thalium Stress Test result(reversable defect), flat slope of the peak exercise
# ST segment are likely to have a low chances of getting heart disease.  
#%%
#Step 3: Data Cleaning

# Remove duplicates
df= df.drop_duplicates()
df.duplicated().sum()

#In this step, only duplicate is removed. 
#The label encoder is not required as the data has already been converted 
#into numeric.

#%%
#Step 4: Features selection
cat_column=['sex','cp','fbs','restecg','exng','slp','thall','output']
con_column=['age','trtbps','chol','thalachh','caa']

#1) Categorical data vs categorical (Output)-Cramer's V
for cat in cat_column:
    print(cat)
    confussion_matrix =pd.crosstab(df[cat], df['output']).to_numpy()
    print(cramers_corrected_stat(confussion_matrix))

#2) Continuous data vs categorical(output)
for con in con_column:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis=-1),df['output'])
    print(lr.score(np.expand_dims(df[con],axis=-1),df['output']))

# Based on Cramer's V and logistic regression, seven(7) features with a score of
# more than 0.5 are selected. 
# The features are: cp(0.509), thall(0.520), age(0.619), trtbps(0.579),
# chol(0.533), thalachh(0.701), and caa(0.738)

X=df.loc[:,['age','cp','trtbps','thall', 'chol', 'thalachh','caa']]
y=df['output']
#%%    
#Step 5: Pre-processing

#1) Train-split test
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                      random_state=123)
#2) Pipelines
# The dataset's problem is the classification, 
# hence, several classifiers are tested in the pipeline analysis

#i) Determine wether MMS or SS is better in this dataset
#ii) Determine which classifier works the best in this dataset

#Step 1: Call the pipeline and create pipeline
#1.LogisticRegression
lr_ss =Pipeline([('Standard Scaler', StandardScaler()),
           ('Logistic Classifier', LogisticRegression())])

lr_mms = Pipeline([('Min Max Scaler', MinMaxScaler()),
         ('Logistic Classifier', LogisticRegression())])

#2.DecisionTreeClassifier
DT_ss = Pipeline([('Standard Scaler', StandardScaler()),
           ('Decision Tree Classifier', DecisionTreeClassifier())])

DT_mms = Pipeline([('Min Max Scaler', MinMaxScaler()),
         ('DecisionTreeClassifier', DecisionTreeClassifier())])

#3.RandomForestClassifier
RF_ss = Pipeline([('Standard Scaler', StandardScaler()),
           ('RandomForestClassifier', RandomForestClassifier())])

RF_mms = Pipeline([('Min Max Scaler', MinMaxScaler()),
         ('RandomForestClassifier', RandomForestClassifier())])

#4.K-Nearest Neighbor
knn_ss = Pipeline([('Standard Scaler', StandardScaler()),
           ('knn', KNeighborsClassifier())])

knn_mms = Pipeline([('Min Max Scaler', MinMaxScaler()),
         ('knn', KNeighborsClassifier())])

#join them, create a list for the pipeline so that you can iterate them
pipelines = [lr_ss,lr_mms,DT_ss,DT_mms,RF_ss, RF_mms,knn_ss,knn_mms]

#fitting of data
for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
pipe_dict={0:'SS+LR', 1: 'MMS+LR', 2:'SS+DT', 3: 'MMS+DT', 4:'SS+RF', 5:'MMS+RF',
           6: 'SS+KNN', 7: 'MMS+KNN'}

best_accuracy=0

# model evaluation
for i, model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test,y_test)> best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]  
        
print('The best scaling approach for heart dataset will be {} with accuracy of {}'.format(best_scaler,best_accuracy))
    
#In the pipeline step,LR + SS appears to be the optimal combination of method 
# for heart dataset with a score of 0.80    
#%% Performance of model
#1) The selected best combination
pipe_lr =Pipeline([('Standard Scaler', StandardScaler()),
            ('LogisticClassifier', LogisticRegression())])

#2) Gridsearch
#Grid_param
#max_iter: [100,500,1000]
#C:np.logspace(-4, 4, 50)
#solver= ['newton-cg', 'lbfgs', 'liblinear']
#penalty = ['none', 'l1', 'l2', 'elasticnet']


grid_param=[{'LogisticClassifier__max_iter': [100,500,1000],
             'LogisticClassifier__C':np.logspace(-4, 4, 50),
             'LogisticClassifier__solver':['newton-cg','lbfgs','liblinear']}]

gridsearch =GridSearchCV(pipe_lr,grid_param,cv=5,verbose=1,n_jobs=-1) 
best_model = gridsearch.fit(X_train,y_train)
print(best_model.score(X_test, y_test))
print(best_model.best_index_)
print(best_model.best_params_)
print(best_model.best_score_)

#Based on the GridSearchCV, the best parameters for this training will be:
# max iteration of 100 with C value of 0.0409 and a liblinear for solver
# which will return best model with a score of 0.8388

#3) Retrain the model with the best parameters

pipe_lr =Pipeline([('Standard Scaler', StandardScaler()),
            ('LogisticClassifier', LogisticRegression(max_iter=100,
                                                      C=0.0409,
                                                      solver='liblinear'))])
            
pipe_lr.fit(X_train,y_train)

pkl_fname = 'best_model.pkl'
with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(best_model,file)
#%% The analysis of the model

y_true=y_test
y_pred=best_model.predict(X_test)
print(classification_report(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Based on the trained data analysis, the model is able to accurately predict
# chances for a person to get the heart disease with a f1 score value of 0.79.
# Thus, the model is saved and deployed for an app to predict the status of 
# getting heart disease