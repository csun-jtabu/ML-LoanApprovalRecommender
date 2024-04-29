# Jaztin Tabunda, Yana Zaynullina, Arohan Pillai
# COMP 542
# Project - Loan Approval Recommendation System using Support Vector Machine
# Prof. Rashida

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk

from sklearn.preprocessing import StandardScaler

from sklearn import svm, metrics

from sklearn.metrics import (precision_score, recall_score, 
f1_score, accuracy_score, make_scorer, confusion_matrix)

from sklearn.model_selection import (train_test_split, StratifiedKFold, 
cross_validate, GridSearchCV, learning_curve)

from sklearn.feature_selection import RFE

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
                                                display_labels = ['Approved', 'Rejected']) # [0, 1]
    
    # we then plot the matrix
    displayMatrix.plot()
    plt.show()
pass

# accuracy when doing k fold cross validation
def kFold(mySVM, X_training, y_training):
    
    # These are the metrics we want to find out about the training data
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
    
    # Define hyperparameters to perform gridSearchCV
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
    
    # We get the column names for print output
    column_names = X.columns
    # We put the training data in a dataframe to allow to be processed
    X_training = pd.DataFrame(X_training)
    
    svc = svm.SVC(kernel="linear") # Finding the best feature for SVM
    
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

def plotLearningCurves(trainedSVM, X_train, y_train):
    # Specifying training sizes, there is a limitation of a maximum size of 2390
    train_sizes = np.linspace(0.1, 1.0, 10) * 2390
    # Converting to integer
    train_sizes = train_sizes.astype(int)
    # Computing the learning curves
    # cv stands for cross-validation folding
    train_sizes_abs, train_scores, test_scores = learning_curve(trainedSVM, X_train, y_train, cv=5, train_sizes=train_sizes)

    # Mean of scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 8))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')

    # Plot settings
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

def main():
    
    # we first convert the dataset from categorical to numerical data
    dataset = convertToNumerical()
    
    # get all values from each column except last column/classifier and
    # first column/column 0 (loan id)
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

    # Plotting learning curves
    plotLearningCurves(trainedSVM, X_train, y_train)
    
    # Generating Confusion Matrix
    generateConfMatrix(trainedSVM, X_test, y_test)
    
    # Launching Loan Approval Recommender System
    applicationGUI(trainedSVM, featureScaler)
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
        
        # we define the input in a dictionary
        submission = {' income_annum': [income], ' loan_amount': [la], ' loan_term': [loanTerm],
                      ' cibil_score': [cs]}
        
        # we put it in a dataframe
        submission = pd.DataFrame(submission)
        
        # we scale the new data point using the same scalar used for the test
        # and training data
        newSubmission = self.featureScaler.transform(submission)
        
        # we make a prediction using the inputted data
        prediction = self.trainedSVM.predict(newSubmission)
        outcome = prediction[0]
        
        # We display if they have been approved or not
        if(outcome == 0):
         self.predictionSubmission.config(text='Approved')    
        else:
         self.predictionSubmission.config(text='Rejected')
    pass
pass

main()