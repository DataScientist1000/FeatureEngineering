# -*- coding: utf-8 -*-

import os
data_path  = os.path.abspath("HandlingImbalancedData/Data/creditcard.csv")

#---------------------------------------------------

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE

#---------------------------------------------------
df = pd.read_csv(data_path)
df.head()
df.columns
df.Class.value_counts()
df.describe()

#Analysing the class distribution
class_dist =  df.Class.value_counts()
print(class_dist)
print("Class 0: {:0.2f}%".format(100 * class_dist[0] / (class_dist[0] + class_dist[1])))
print("Class 1: {:0.2f}%".format(100 * class_dist[1] / (class_dist[0] + class_dist[1])))

#split the data
X = df.drop(columns = ['Time','Class'])
y = df['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,test_size =0.3, stratify = y)
#Logistic regreesion Model

model = LogisticRegression()
model.fit(X_train,y_train)
y_Pred = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test,y_Pred))
print(classification_report(y_test,y_Pred))
print(confusion_matrix(y_test,y_Pred))
sns.heatmap(confusion_matrix(y_test,y_Pred),annot= True,fmt='.2g')


print("Before overSampling , count of label 1: {} ".format(sum(y_train == 1)))
print("Before overSampling , count of label 0: {} ".format(sum(y_train == 0)))
#SMOTE
"""
numpy.ravel(array, order = ‘C’) : returns contiguous flattened array(1D array with all the input-array elements and with the same type as it). A copy is made only if needed.
Parameters :

array : [array_like]Input array. 
order : [C-contiguous, F-contiguous, A-contiguous; optional]         
         C-contiguous order in memory(last index varies the fastest)
         C order means that operating row-rise on the array will be slightly quicker
         FORTRAN-contiguous order in memory (first index varies the fastest).
         F order means that column-wise operations will be faster. 
         ‘A’ means to read / write the elements in Fortran-like index order if,
         array is Fortran contiguous in memory, C-like order otherwise
"""
sm = SMOTE(sampling_strategy = 0.5, k_neighbors = 5,random_state=100)
X_train_res , y_train_res = sm.fit_sample(X_train,y_train.ravel())

print("After Oversampling , the shape of the train_x : {}".format(X_train_res.shape))
print("After Oversampling , the shape of the train_y : {}".format(y_train_res.shape))

print("After overSampling , count of label 1: {} ".format(sum(y_train_res == 1)))
print("After overSampling , count of label 0: {} ".format(sum(y_train_res == 0)))

#apply the model again
lr = LogisticRegression()
lr.fit(X_train_res,y_train_res)
predictions = lr.predict(X_test)

print("Accuracy ",accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions),annot= True,fmt='.2g')

"""
Summary: 
    After SMOTE recall we have increased the 0.86 before it was 0.61
"""
