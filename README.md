![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)


# Project Title
The prediction of heart disease. 

## Description
In this project, spyder software was used to train the dataset in predicting the chances for a person to get the heart disease. 
After the training has completed, the trained data was deployed in streamlit to create a predicting app which consisted of a few important variables.

## Step-by-step
The dataset named "heart" contained about 400 number of samples which consisted of 14 variables. The exploratory data analysis was conducted to filter out the best parameters which are important in predicting the likeliness for a person to have a heart disease.
In the feature selections, logistic regression and Cramer's V analysis were conducted. 7 parameters were chosen as they have higher correlations value (>0.5) against the output variable.
After filtering out the variable, the pre-processing step was conducted and the best combination was achieved by using pipeline analysis.
![Best combination for pipeline](https://user-images.githubusercontent.com/107612253/174921219-c56b9654-5214-4162-baec-b21bd55f273b.png)

The dataset was trained using logistic classifier and Standard Scaler which return a relatively good result for a predictive model with a f1 score of 0.79.
![Model analysis(F1)](https://user-images.githubusercontent.com/107612253/174921013-78b2b271-de80-4486-85e0-590f0cbd5a5d.png)

## Deployment
The trained model was deployed and visualized in an app using Streamlit.
![streamlit](https://user-images.githubusercontent.com/107612253/174794990-5dbb07b7-ffc5-42f3-b412-f9589840d734.png)
![streamlit_2](https://user-images.githubusercontent.com/107612253/174795262-7591579e-e783-4734-b0d2-386d9b5b3ed6.png)

## Acknowledgement
I would like to sincerely thank my Data Analytics course's trainer, Mr. Warren Loo and Mr.Rashik Rahman for the dataset which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset).
