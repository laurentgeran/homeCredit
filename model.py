import os
import time
import gc
from tqdm import tqdm
import datetime
from dateutil import relativedelta
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import statistics
import scipy
import random
import math
import sklearn
import re

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Preprocess
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder 
from category_encoders.leave_one_out import LeaveOneOutEncoder
from sklearn.preprocessing import RobustScaler

# Sampling
from imblearn.over_sampling import SMOTE 

# Viz
from sklearn.decomposition import PCA

# Modeling
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

# Hyperparameters 
from skopt.space import Integer
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
 
# Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Feature selection
from boruta import BorutaPy
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

# Interpretation
import shap

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
    print(np.mean(cv["test_auc"]))
    print(np.mean(cv["test_accuracy"]))
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
feat_selector = BorutaPy(LGBMClassifier(num_boost_round = 100), n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(under_X.values, under_y.values)

# oversampling using feature selection
over_X=applicationTrain[applicationTrain_X.columns[feat_selector.support_]]
model = modeling(LGBMClassifier(), over_X, applicationTrain_y, overSampling = True)

with open('clf_feat_over.pkl', 'wb') as output_file:
    pickle.dump(model, output_file)