# -*- coding: utf-8 -*-
"""
Applying Random Forest on unreal data
#it knows what feature is contributing the best, allows to understand feature importance
"""
#on unreal data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"D:\Anum\Learning ML DL\5-Random Forest\Images_Analyzed_Productivity.csv")
print(df.head())

#check if the dataset is balanced
sizes = df['Productivity'].value_counts(sort =1)
print(sizes)

#drop extra columns, not needed!
df.drop(['Images_Analyzed'],axis=1,inplace=True)
df.head()
df.drop(['User'],axis=1,inplace=True)
df.head()

#make sure you dont have any missing values, drop values you dont require
df = df.dropna()
df.head()

#convert non-numeric data into numeric
df.Productivity[df.Productivity == 'Good '] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
df.head()

#Define dependent variables
Y = df['Productivity'].values #it is an ndarray object, but we need Y to be int 
#convert it into int
Y = Y.astype('int')


Define independent variables
X = df.drop(labels=['Productivity'],axis=1)
X.head()

#Now apply Random Forest 
from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.1, random_state=20)
#random_state keeps the dataset split same

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, random_state=20)
model.fit(Xtrain,Ytrain)

prediction_test = model.predict(Xtest)
print(prediction_test)

from sklearn import metrics

acc = metrics.accuracy_score(Ytest, prediction_test)
print('Accuracy: ', acc)


feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index = feature_list).sort_values(ascending = False)
print(feature_imp)
