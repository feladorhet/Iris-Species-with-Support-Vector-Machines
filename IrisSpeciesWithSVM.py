import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris = sns.load_dataset("iris")
#####['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']

##EXPLORATORY DATA ANALYSIS
sns.pairplot(data=iris, hue="species", palette="coolwarm")
sns.kdeplot(data=iris[iris["species"] == "setosa"], x="sepal_width", y="sepal_length", cmap="plasma", fill=True)
#plt.show()

X = iris.drop("species", axis=1)
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#THE VERY FIRST UNOPTIMIZED SUPPORT VECTOR MACHINE INITIALIZATION AND TESTING 
svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_preds = svc_model.predict(X_test)
print("UNOPTIMIZED SVM OUTCOMES:"+"\n")
print(confusion_matrix(y_true = y_test, y_pred = svc_preds))
print(classification_report(y_true = y_test, y_pred = svc_preds)+"\n")

##NOW CREATE A BETTER MODEL BY FINDING OPTIMAL VALUES FOR SVC PARAMETERS
params = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']}
print("RUNNING A GRID SEARCH TO FIND OPTIMAL PARAMETERS"+"\n")
grid = GridSearchCV(estimator=SVC(), param_grid=params, verbose=5)
grid.fit(X_train, y_train)
grid_preds = grid.predict(X_test)
print("OPTIMIZED SVM OUTCOMES"+"\n")
print(confusion_matrix(y_true = y_test, y_pred = grid_preds))
print(classification_report(y_true = y_test, y_pred = grid_preds))

