import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from mlflow import log_metric, log_param, log_artifacts, log_metrics

if __name__ == '__main__':
    print('Starting the experiment')
    
    ##mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name = 'titanic_mlflow')

    mlflow.autolog()

    df = pd.read_csv('Titanic+Data+Set.csv')

    #sex = pd.get_dummies(df['Sex'],drop_first = True)
    #embark = pd.get_dummies(df['Embarked'],drop_first = True)

    df['Sex']=(df['Sex']=='male').astype(int)

    df = df.drop(columns=['Cabin','Name','Ticket','PassengerId'],axis=1)

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df = pd.get_dummies(df, columns=['Embarked'])

    X_train,X_test,y_train,y_test = train_test_split(df.drop('Survived',axis=1),
                                                 df['Survived'],test_size=0.30,random_state = 101)
    print(X_train.shape, X_test.shape)
    log_param("Train shape",X_train.shape )

    ## Hyperparameter tuning
    model = RandomForestClassifier()

    ## Parameter Search shape
    params = [{'criterion': ['entropy', 'gini'],
                'n_estimators': [10,30,50,70,90],
                'max_features': ['sqrt', 'log'],
                'max_depth': [5,10,15],
                'min_samples_split': [2,5,8,11],
                'min_samples_leaf': [1,5,9],
                'max_leaf_nodes': [2,5,8,11]}]
    
    ## Cross validation
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    ## GridSearch
    tuning = GridSearchCV(estimator=model, cv=cv, scoring='accuracy', param_grid=params)

    ## Train and optimize the estimator
    tuning.fit(X_train, y_train)

    ## Best parameters found
    print('Best Parametyers found using GridSearch:', tuning.best_params_)

    train_accuracy = model.score(X_train, y_train)  # performance on train data
    test_accuracy = model.score(X_test, y_test)  # performance on test data

    ## Logging metrics
    log_metric("Accuracy for this run", test_accuracy)
    mlflow.sklearn.log_model(model, "RandomForestModel")
