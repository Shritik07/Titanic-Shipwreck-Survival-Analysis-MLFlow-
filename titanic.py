import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
import mlflow
import mlflow.sklearn
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

    #Training the Decision Tree Classifier Model

    dtmodel = DecisionTreeClassifier(criterion = "gini",
                               max_depth=8, min_samples_leaf=5)

    dtmodel.fit(X_train, y_train)
    print("Model trained")

    train_accuracy = dtmodel.score(X_train, y_train)  # performance on train data
    test_accuracy = dtmodel.score(X_test, y_test)  # performance on test data
    
    log_metric("Accuracy for this run", test_accuracy)
    
    mlflow.sklearn.log_model(dtmodel, "DecisionTreeModel")
    #mlflow.log_artifact('Titanic+Data.csv')

