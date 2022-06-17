import pandas as pd
import numpy as np

# Preprocess
from scipy.stats.mstats import winsorize

# Sampling
from imblearn.over_sampling import SMOTE 

# Modeling
from lightgbm import LGBMClassifier

# Hyperparameters 
from skopt.utils import use_named_args
from skopt import gp_minimize
 
# Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Feature selection
from boruta import BorutaPy

# Serialization
import pickle

# myFunctions

def splitData(X,y):
    XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size=0.2)
    return XTrain, XTest, yTrain, yTest

def modeling(classifier, X, y, featuresSelected:list= ["All"], overSampling = False):
    
    X_=X
    if featuresSelected[0] != "All":
        X_=X_.loc[:,featuresSelected]
    XTrain, XTest, yTrain, yTest=splitData(X_,y)
    if overSampling:
        oversampling = SMOTE()
        XTrain, yTrain = oversampling.fit_resample(XTrain, yTrain)
    scoring = {"auc": make_scorer(roc_auc_score), "accuracy": make_scorer(accuracy_score)}
    cv = cross_validate(classifier,XTrain, yTrain , cv=4,scoring = scoring)
    #print(np.mean(cv["test_auc"]))
    #print(np.mean(cv["test_accuracy"]))
    classifier.fit(XTrain, yTrain,eval_metric='auc', eval_set=[(XTest, yTest)])
    yPred=classifier.predict(XTest)
    print("classification report : ", classification_report(yTest,yPred))

    return classifier

applicationTrain=pd.read_csv('applicationTrain.csv', index_col = 0)
applicationTrain_y=np.ravel(pd.read_csv('applicationTrain_y.csv', index_col = 0))
applicationTrain_X=pd.read_csv('applicationTrain_X.csv', index_col = 0).drop(columns=["SK_ID_CURR"])

model = modeling(LGBMClassifier(), applicationTrain_X, applicationTrain_y)

with open('clf.pkl', 'wb') as output_file:
    pickle.dump(model, output_file)

# random under sampling
n=len(applicationTrain[applicationTrain["TARGET"]==1])
dataBaseline=applicationTrain[applicationTrain["TARGET"]==0].sample(n,random_state=0,axis=0)
dataBaseline=dataBaseline.append(applicationTrain[applicationTrain["TARGET"]==1])
under_y=dataBaseline["TARGET"]
under_X=dataBaseline.drop(columns=["TARGET","SK_ID_CURR"])

# feature selection with BorutaPy
feat_selector = BorutaPy(LGBMClassifier(num_boost_round = 100), n_estimators='auto', random_state=1)
feat_selector.fit(under_X.values, under_y.values)

# oversampling using feature selection
featSelect_X=applicationTrain[applicationTrain_X.columns[feat_selector.support_]]
featSelect_X.to_csv('featSelectTrain_X.csv')
modelOver = modeling(LGBMClassifier(), featSelect_X, applicationTrain_y, overSampling = True)

# Optimisation des hyperparamètres

with open('clf_feat_over.pkl', 'wb') as output_file:
    pickle.dump(modelOver, output_file)