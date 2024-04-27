# Jaztin Tabunda, Yana Zaynullina, Arohan Pillai
# COMP 542
# Project - Loan Approval Recommendation System using Support Vector Machine
# Prof. Rashida

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn import metrics

import tkinter as tk

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

# initial accuracy
def getAccuracy(trainedSVM, X_test, y_test):
    # .predict basically takes in your test values and then
    predictions = trainedSVM.predict(X_test)
    
    # This is how we get the accuracy of each model/SVM
    # Essentially, this library is doing 
    # total number of correct predictions / total number of predictions
    # to calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy
pass

def getPrecision(trainedSVM, X_test, y_test):
    # .predict basically takes in your test values and then
    predictions = trainedSVM.predict(X_test)
    
    # This is how we get the Precision of each model/SVM
    # Essentially, this library is doing 
    # TP / TP + FP
    precision = precision_score(y_test, predictions)
    
    return precision
pass

def getRecall(trainedSVM, X_test, y_test):
    # .predict basically takes in your test values and then
    predictions = trainedSVM.predict(X_test)
    
    # This is how we get the Precision of each model/SVM
    # Essentially, this library is doing 
    # TP / TP + FN
    recall = recall_score(y_test, predictions)
    
    return recall
pass

def getF1Score(trainedSVM, X_test, y_test):
    # .predict basically takes in your test values and then
    predictions = trainedSVM.predict(X_test)
    
    # This is how we get the Precision of each model/SVM
    # Essentially, this library is doing 
    # (2)(precision)(recall) / (precision + recall)
    f1Score = f1_score(y_test, predictions)
    
    return f1Score
pass

def generateConfMatrix(trainedSVM, X_test, y_test):
    # .predict basically takes in your test values and then
    predictions = trainedSVM.predict(X_test)
    
    # we generate the confusion matrix by putting in the actual values
    # and the predicted values
    matrix = confusion_matrix(y_test, predictions)
    
    # We then pass it into the sklearn matrix method
    displayMatrix = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, 
                                                display_labels = [0, 1]) # [0, 1]
    
    # we then plot the matrix
    displayMatrix.plot()
    plt.show()
pass

# accuracy when doing k fold cross validation
def kFold(mySVM, X_training, y_training):
    
    # # Define a Pipeline with StandardScaler and SVM
    # pipeline = Pipeline([('scaler', scaler), ('svm', mySVM)])
    
    scoring = { 'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score), 'f1': make_scorer(f1_score) }
    
    # we create the k fold object and split the dataset into 10
    kFolder = StratifiedKFold(n_splits=10)

    # we pass in the pipeline which will scale our data and fit it to the SVM
    # then it will perform Stratified K Fold
    results = cross_validate(mySVM, X_training, y_training, cv=kFolder, scoring=scoring)
    
    return results
pass

# This method will be used to find the best hyperparameters from a given set
def findingHyperparam(mySVM, X_training, y_training):
    
    # # we use pipeline to group the items we'll be using to process the
    # # gridSearchCV. It will transform the data using scaler then pass it in
    # # our model
    # pipeline = Pipeline([('scaler', scaler), ('svm', mySVM)])
    
    # # Define hyperparameters to search
    # gridParameters = {
    #     'svm__C': [0.1, 1, 10, 100, 1000, 10000],          # Regularization parameter
    #     'svm__kernel': ['rbf'],   # Kernel type 'linear', 'rbf', 'poly'
    #     'svm__gamma': [0.001, 0.0055,0.01, 0.1, 1, 10]     # Kernel coefficient for 'rbf' and 'poly'
    # }
    
    # Define hyperparameters to search
    gridParameters = {
        'C': [0.1, 1, 10, 100, 1000, 10000],          # Regularization parameter
        'kernel': ['rbf'],   # Kernel type 'linear', 'rbf', 'poly'
        'gamma': [0.001, 0.0055,0.01, 0.1, 1, 10]     # Kernel coefficient for 'rbf' and 'poly'
    }
    
    # We pass in the pipeline which contains the scaler, the svm and parameters
    gridSearch = GridSearchCV(mySVM, gridParameters, cv=10)
    
    # This is to train/start the gridSearch with our data
    gridSearch.fit(X_training, y_training)
    
    # we put the results in a pandas dataframe which
    results = pd.DataFrame(gridSearch.cv_results_)
    
    # we make sure the dataframe displays all columns...
    pd.set_option('display.max_columns', None)
    # all columns in this array
    columnsSelected = results[['params', 'mean_test_score', 'rank_test_score']]
    
    # Print the DataFrame
    print(columnsSelected)
pass

def findingBestFeatures(X, X_training, y_training):
    
    # # We scale the data using standardscaler so we reduce the feature biases
    # # in which different features have different scales in data
    # scaler = StandardScaler()
    column_names = X.columns
    X_training = pd.DataFrame(X_training)
    
    # # We scale the entire feature set using StandardScaler
    # X_scaled = scaler.fit_transform(X)
    
    svc = SVC(kernel="linear") # Finding the best feature for SVM
    
    # We use Recursive Feature Elimination to find the rank of which feature
    # We want a 1 ranking to 1 feature, so n_features.. = 1
    # RFE will remove 1 feature per iteration, so step = 1
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    # We train/start the RFE model
    rfe.fit(X_training, y_training)
    
    # we get the ranking list
    ranking = rfe.ranking_
    # we sort the ranking from least to greatest
    sortedRanking = np.argsort(ranking)
    
    # we print the ranking
    print("Feature ranking:")
    for id in sortedRanking:
        print(f"{column_names[id]}: Rank {ranking[id]}")
    print('\n')
pass

def main():
    
    # we first convert the dataset from categorical to numerical data
    dataset = convertToNumerical()
    
    # get all values from each column except last column/classifier and
    # first column (loan id)
    X = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    
    # get all values from only the last column/classifier
    y = dataset.iloc[:, -1].values
    
    # This is how we split the dataset into different sets Training and Testing Data. 
    # We put a random_state to make sure our data is consistent throughout the program
    # best accuracy is with test_size=0.30
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.30, random_state=1)

    # we do one of the common ways to do feature scaling (standard scaling)
    featureScaler = StandardScaler()
    featureScaler.fit(X_train)  # this is how we train the scaler with the training data to scale
    X_train = featureScaler.transform(X_train) # we scale the training data separate from the test data
    X_test = featureScaler.transform(X_test) # we scale the test data separate from the training data    

    # we find the best feature using Recursive Feature Extraction (RFE)
    findingBestFeatures(X, X_train, y_train)

    # According to RFE, we remove the least important features first
    # We remove the least important features/column indices: 
    
    # removed luxury asset value (10)
    # removed no of dependents (1) - This results in a drop in accuracy but we continue hyst in case
    # removed residential assets value (8)
    # removed education (2)
    # removed commercial assets value (9)
    # removed self employed (3)
    # removed bank asset value (11)
    
    # Any more removed results in worse accuracy
    X = dataset.iloc[:, [4, 5, 6, 7]] 
    
    # We split and rescale again with new feature set
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.30, random_state=1)

    featureScaler = StandardScaler()
    featureScaler.fit(X_train)  
    X_train = featureScaler.transform(X_train) 
    X_test = featureScaler.transform(X_test) 
    
    
    # The default settings: C=1.0, kernel='rbf', gamma='scale'
    mySVM = svm.SVC(C=1.0, kernel='rbf', gamma='scale') 
    
    # we get the accuracy of the model using kfold
    # a little bit less accurate but it's because it's doing cross validation
    results = kFold(mySVM, X_train, y_train)
    
    kFoldAccuracy = results['test_accuracy']
    kFoldPrecision = results['test_precision']
    kFoldRecall = results['test_recall']
    kFoldF1 = results['test_f1']
    
    print("K = " + str(len(kFoldAccuracy)))
    print("K Fold Cross Val Scores = " + str(kFoldAccuracy))
    print("K Fold Accuracy with default settings = " + str(kFoldAccuracy.mean()))
    print("K Fold Precision Scores: " + str(kFoldPrecision))
    print("K Fold Precision with default settings = " + str(kFoldPrecision.mean()))
    print("K Fold Recall Scores: " + str(kFoldRecall))
    print("K Fold Recall with default settings = " + str(kFoldRecall.mean()))
    print("K Fold F1 Scores: " + str(kFoldF1))
    print("K Fold F1 Score with default settings = " + str(kFoldF1.mean()))
    print('\n')
    
    # Now we find the hyperparameters
    findingHyperparam(mySVM, X_train, y_train)
    
    # We find that the best hyperparameters are C = 10000 and gamma = 0.1
    mySVM = svm.SVC(C=10000, kernel='rbf', gamma=0.1)
    
    # .fit basically trains the model again
    trainedSVM = mySVM.fit(X_train, y_train)
    
    # we get the accuracy metrics of the model by just training it with 
    # best settings and the data has been scaled
    accuracy = getAccuracy(trainedSVM, X_train, y_train)
    precision = getPrecision(trainedSVM, X_test, y_test)
    recall = getRecall(trainedSVM, X_test, y_test)
    f1Score = getF1Score(trainedSVM, X_test, y_test)
    print("This is the Training data accuracy with default settings and no k fold: " 
          + str(accuracy))
    accuracy = getAccuracy(trainedSVM, X_test, y_test)
    print("This is the Test data accuracy with Best settings and no k fold: " 
          + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1Score))
    print('\n')
    
    generateConfMatrix(trainedSVM, X_test, y_test)
    
    app = applicationGUI(trainedSVM, featureScaler)
    
pass

class applicationGUI:
    def __init__(self, trainedSVM, featureScaler):
        self.trainedSVM = trainedSVM
        self.featureScaler = featureScaler
        self.root = tk.Tk()
        self.root.geometry("800x500")
        self.root.title("Loan Approval Application")
        self.createWidgets()
        self.formatWidgets()
        self.root.mainloop()
    pass

    def createWidgets(self):
        self.creditScore = tk.StringVar()
        self.loanAmount = tk.StringVar()
        self.income = tk.StringVar() #self.root
        self.loanTerm = tk.StringVar() # in years
        

        self.cs_label = tk.Label(self.root, text="Credit Score:")
        self.cs_entry = tk.Entry(self.root, width=40, textvariable=self.creditScore)

        self.la_label = tk.Label(self.root, text="Loan Amount:")
        self.la_entry = tk.Entry(self.root, width=40, textvariable=self.loanAmount)

        self.income_label = tk.Label(self.root, text="Income:")
        self.income_entry = tk.Entry(self.root, width=40, textvariable=self.income)


        self.loanTerm_label = tk.Label(self.root, text="Loan Term in years:")
        self.loanTerm_entry = tk.Entry(self.root, width=40, textvariable=self.loanTerm)
        
        self.prediction_label = tk.Label(self.root, text="Prediction:")
        self.predictionSubmission = tk.Label(self.root, text="")
        
        # Create a button to submit information
        self.submit_button = tk.Button(self.root, text="Submit", command=self.submitInfo)
        
    pass

    def formatWidgets(self):
        self.cs_label.grid(row=0, column=0, padx=10, pady=5)
        self.cs_entry.grid(row=0, column=1, padx=10, pady=5)
        self.la_label.grid(row=1, column=0, padx=10, pady=5)
        self.la_entry.grid(row=1, column=1, padx=10, pady=5)
        self.income_label.grid(row=2, column=0, padx=10, pady=5)
        self.income_entry.grid(row=2, column=1, padx=10, pady=5)
        self.loanTerm_label.grid(row=3, column=0, padx=10, pady=5)
        self.loanTerm_entry.grid(row=3, column=1, padx=10, pady=5)
        self.prediction_label.grid(row=4, column=0, padx=10, pady=5)
        self.predictionSubmission.grid(row=4, column=1, padx=10, pady=5)
        self.submit_button.grid(row=5, columnspan=2, padx=10, pady=10)
    pass

    def submitInfo(self):
        cs = int(self.creditScore.get())
        la = int(self.loanAmount.get())
        income = int(self.income.get())
        loanTerm = int(self.loanTerm.get())
        
        submission = np.array([[income, la, loanTerm, cs]])
        newSubmission = self.featureScaler.transform(submission)
        
        prediction = self.trainedSVM.predict(newSubmission)
        outcome = prediction[0]
        print(prediction)
        if(outcome == 0):
         self.predictionSubmission.config(text='Approved')    
        else:
         self.predictionSubmission.config(text='Rejected')
    pass

pass

main()