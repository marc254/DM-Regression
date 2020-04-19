# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:50:46 2020

@author: skambou
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.contingency import chi2_contingency
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import warnings

## Part 1 Regression
### Data Description
#### Rows: 21613
#### Columns : 21

df = pd.read_csv('E:\\Marc\\Data Mining\\Project\\Regression\\kc_house_data.csv')
df.info()
df.isnull().sum()
df.columns


# extract independents variables; has to be a 2D array
X = df.iloc[:,3:]
y = df['price']

#splitting data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)


sns.heatmap(X)

#Scale the data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

#import linear model ; Ordinary Least Square Regression
import statsmodels.api as sm
X2_train = sm.add_constant(X_train_scaled)
X2_test = sm.add_constant(X_test_scaled)
ols = sm.OLS(y_train,X2_train)
lr = ols.fit()

while lr.pvalues.max()>0.05:
    X2_train=np.delete(X2_train,lr.pvalues.argmax(),axis=1)
    X2_test=np.delete(X2_test,lr.pvalues.argmax(),axis=1)
    ols = sm.OLS(y_train,X2_train)
    lr = ols.fit()

print(lr.summary())



#create an instance of the model
model = LinearRegression()



#train the model
model.fit(X_train_scaled, y_train)


#coefficient of determination
model.score(X_train_scaled,y_train)
model.score(X_test_scaled,y_test)


#use model to predict
y_pred_test = model.predict(X_test_scaled)

y_pred_train = model.predict(X_train_scaled)


from sklearn.metrics import r2_score, mean_squared_error
test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_pred_test)))
test_set_r2_test = r2_score(y_test, y_pred_test)
test_set_r2_train = r2_score(y_train, y_pred_train)
print(test_set_rmse, test_set_r2_test, test_set_r2_train)



#preprocessing
X_scaled = sc.fit_transform(X)

# Perform 4-fold cross validation
scores = cross_val_score(model, X_scaled, y, cv=4)
print ('Cross-validated scores:', scores)


# Correlation Matrix

features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
            'zipcode','lat','long','sqft_living15','sqft_lot15']

mask = np.zeros_like(df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9})