# coding: utf-8
# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

# Imports
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import sqlite3
from timeit import default_timer as timer

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from m110_feat_extraction_pipeline import pipe_transforms
from src.xgb_output import xgb_output
from src.xgb_train_cv import xgb_train_cv

# Start timer
start_time = timer()

# Set random seed for reproducibility
np.random.seed(1337)

# Globals
cache_file = './cache.db'
db_conn    = sqlite3.connect("./inputs/inputs.db")
# Note: we should use try/finally to properly close the DB even if there are exceptions
#       but let's have faith in Python GC

# Import data
df_train = pd.read_sql_query("select SK_ID_CURR, TARGET from application_train;", db_conn)
print('Input training data has shape: ',df_train.shape)

X = df_train[['SK_ID_CURR']]
y = df_train[['TARGET']]
del df_train

X_test = pd.read_sql_query("select SK_ID_CURR from application_test;", db_conn)

# Create folds
cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=1337)
folds = list(cv.split(X,y))

# Pipeline processing
X, X_test, _, _, _, _ = pipe_transforms(X, X_test, y, db_conn, folds, cache_file)
print('After preprocessing data shape is: ', X.shape)

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

# Quick validation to get a unique name
x_trn, x_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and validate
print("############ Validation, Cross-Validation and Final Classifier ######################")
clf, metric, n_stop = xgb_train_cv(x_trn, x_val, y_trn, y_val, X, y, xgb_params, folds)

# Output
xgb_output(X_test, X_test['SK_ID_CURR'], clf, n_stop, metric)

# Cleanup
db_conn.close()

end_time = timer()
print("################## Success #########################")
print("Elapsed time: %s" % (end_time - start_time))
