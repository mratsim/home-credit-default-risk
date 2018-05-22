# coding: utf-8
# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

# Imports
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import sqlite3
import time
import os
from timeit import default_timer as timer

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from m110_feat_engineering_pipeline import pipe_transforms
from src.xgb_processing import xgb_validate, xgb_cross_val, xgb_output
from src.instrumentation import setup_logs

# Start timer
start_time = timer()

# Log
str_timerun = time.strftime("%Y-%m-%d_%H%M")
tmp_logfile = os.path.join('./outputs/', f'{str_timerun}--run-in-progress.log')
logger = setup_logs(tmp_logfile)

# Set random seed for reproducibility
np.random.seed(1337)

# Globals
cache_file = './cache.db'
db_conn    = sqlite3.connect("./inputs/inputs.db")
# Note: we should use try/finally to properly close the DB even if there are exceptions
#       but let's have faith in Python GC

# Optimize the db connection, don't forget to add the proper indexes as well
db_conn('PRAGMA temp_store = MEMORY;')
db_conn(f'PRAGMA cache_size = {1 << 18};') # Page_size = 4096, Cache = 4096 * 2^18 = 1 073 741 824 Bytes

# Import data
df_train = pd.read_sql_query("select SK_ID_CURR, TARGET from application_train order by SK_ID_CURR;", db_conn)
logger.info(f'Input training data has shape: {df_train.shape}')

X = df_train[['SK_ID_CURR']]
y = df_train[['TARGET']]
del df_train

X_test = pd.read_sql_query("select SK_ID_CURR from application_test order by SK_ID_CURR;", db_conn)

# Create folds
cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=1337)
folds = list(cv.split(X,y))

# Pipeline processing
logger.info("   ===> Preprocessing")
X, X_test, _, _, _, _ = pipe_transforms(X, X_test, y, db_conn, folds, cache_file)
logger.info(f'After preprocessing data shape is: {X.shape}')

end_time = timer()
logger.info("Preprocessing time: %s" % (end_time - start_time))

##############################
# Setup basic XGBoost and validation
# Validation is used to get an unique name only
# Model performance will be measured by proper Cross-Validation

xgb_params                     = {}
xgb_params['seed']             = 1337
xgb_params['objective']        = 'binary:logistic'
xgb_params['eta']              = 0.05
xgb_params['max_depth']        = 4
xgb_params['silent']           = 1
xgb_params['eval_metric']      = "auc"
xgb_params['min_child_weight'] = 1
xgb_params['subsample']        = 0.7
xgb_params['colsample_bytree'] = 0.7
xgb_params['tree_method']      = 'gpu_hist'
xgb_params['grow_policy']      = 'depthwise'

xgb_params = list(xgb_params.items())

###############################

# Quick validation to get a unique name
logger.info("   ===> Validation")
x_trn, x_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
val_score = xgb_validate(x_trn, x_val, y_trn, y_val, xgb_params, seed_val = 0)

# Cross validation
logger.info("   ===> Cross-Validation")
n_stop = xgb_cross_val(xgb_params, X, y, folds)
n_stop = np.int(n_stop * 1.1) # Full dataset is 25% bigger, so we want a bit of leeway on stopping round to avoid overfitting.

# Training
logger.info("   ===> Training")
xgtrain = xgb.DMatrix(X, y)
classifier = xgb.train(xgb_params, xgtrain, n_stop)

# Output
logger.info("   ===> Start predictions")
xgb_output(X_test, X_test['SK_ID_CURR'], classifier, n_stop, val_score)

# Cleanup
db_conn.close()

end_time = timer()
logger.info("   ===>  Success")
logger.info("         Total elapsed time: %s" % (end_time - start_time))
logging.shutdown()

final_logfile = os.path.join('./outputs/', f'{str_timerun}---valid{val_score:.4f}.log')
os.rename(tmp_logfile, final_logfile)
