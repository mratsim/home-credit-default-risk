# coding: utf-8
# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

# Imports
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.xgb_output import xgb_output
from src.xgb_train_cv import xgb_train_cv

# Import data
df_train = pd.read_csv(open("./inputs/application_train.csv", "r"))
print('Input training data has shape: ',df_train.shape)

X = df_train.drop(columns=['TARGET'])
y = df_train['TARGET']
del df_train

# Encode categorical features
categoricals = [feat for feat in X.columns if X[feat].dtype == 'object']
print('Categorical columns are: ', categoricals)

encoders = list()
for feat in categoricals:
  le = LabelEncoder()
  X[feat] = le.fit_transform(X[feat].astype('str'))
  encoders.append(le)

##############################
# Setup basic XGBoost and validation
# Validation is used to get an unique name only
# Model performance will be measured by proper Cross-Validation

xgb_params                     = {}
xgb_params['objective']        = 'binary:logistic'
xgb_params['eta']              = 0.1
xgb_params['max_depth']        = 4
xgb_params['silent']           = 1
xgb_params['eval_metric']      = "auc"
xgb_params['min_child_weight'] = 1
xgb_params['subsample']        = 0.7
xgb_params['colsample_bytree'] = 0.5
xgb_params['seed']             = 1337
xgb_params['tree_method']      = 'gpu_hist'

xgb_params = list(xgb_params.items())

###############################
# Create folds

cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=1337)
folds = list(cv.split(X,y))

# Quick validation to get a unique name
x_trn, x_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and validate
print("############ Validation, Cross-Validation and Final Classifier ######################")
clf, metric, n_stop = xgb_train_cv(x_trn, x_val, y_trn, y_val, X, y, xgb_params, folds)

# Output
df_test = pd.read_csv(open("./inputs/application_test.csv", "r"))
for feat, encoder in zip(categoricals, encoders):
    df_test[feat] = encoder.transform(df_test[feat].astype('str'))

xgb_output(df_test, df_test['SK_ID_CURR'], clf, n_stop, metric)
