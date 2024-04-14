# Jaztin Tabunda, Yana Zaynullina, Arohan Pillai
# COMP 542
# Project - Loan Approval Recommendation System using Support Vector Machine
# Prof. Rashida

import matplotlib.pyplot as matPlot
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

# we pre process the data
def convertToNumerical():
    # we read data from CSV file
    dataFrame = pd.read_csv('loan_approval_dataset.csv')
    # we basically provide numerical values for the categorical data
    # and we update the original dataFrame (inplace arg)
    dataFrame[' education'].replace([' Not Graduate', ' Graduate'], [0, 1], inplace=True)
    dataFrame[' self_employed'].replace([' No', ' Yes'], [0, 1], inplace=True)
    dataFrame[' loan_status'].replace([' Approved', ' Rejected'], 
                              [0, 1], inplace=True)
    
    return dataFrame
pass

def buildSVM(dataset):
    
    # get all values from each column except last column/classifier 
    X = dataset.iloc[:, 1:-1].values
    
    # get all values from only the last column/classifier
    y = dataset.iloc[:, -1].values

    # This is how we split the dataset into different sets Training and Testing Data. 
    # We put a random_state to make sure our data is consistent throughout the program
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.30, random_state=1)
    
    # this makes the SVM Model
    mySVM = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
    
    # .fit basically trains the model
    mySVM = mySVM.fit(X_train, y_train)
    
    # .predict basically takes in your test values and then
    predictions = mySVM.predict(X_test)
    
    # This is how we get the accuracy of each model/SVM
    # Essentially, this library is doing 
    # total number of correct predictions / total number of predictions
    # to calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy
    
pass

dataset = convertToNumerical()
accuracy = buildSVM(dataset)
print(accuracy)