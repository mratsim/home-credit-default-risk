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

from src.cross_val_xgb import cross_val_xgb

# Import data

df_train = pd.read_csv(open("./inputs/application_train.csv", "r"))
print('Input training data has shape: ',df_train.shape)

X = df_train.drop(columns=['TARGET'])
y = df_train['TARGET']
del df_train

# Categorical features
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

def _clf_xgb(x_trn, x_val, y_trn, y_val, seed_val=0, num_rounds=4096):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 4
    param['silent'] = 1
    param['eval_metric'] = "auc"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.5
    param['seed'] = seed_val
    param['tree_method'] = 'gpu_hist'
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(x_trn, label=y_trn)
    xgtest = xgb.DMatrix(x_val, label=y_val)

    print('Start Validation training on 80% of the dataset...')
    # train
    watchlist = [ (xgtest, 'test') ]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
    print('End trainind on 80% of the dataset...')
    print('Start validating prediction on 20% unseen data')
    # predict
    y_pred = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

    return model, y_pred, plst, model.best_ntree_limit

def train(x_trn, x_val, y_trn, y_val, X_train, y_train, folds):

    clf, y_pred, params, n_stop = _clf_xgb(x_trn, x_val, y_trn, y_val)

    # eval
    metric = roc_auc_score(y_val, y_pred)
    print('\n\nWe stopped at boosting round:  ', n_stop)
    print('The ROC AUC of prediction is: ', metric)

    xgtrain = xgb.DMatrix(X_train, label=y_train)

    print('\n\nCross validating Stratified 7-fold... and retrieving best stopping round')

    mean_round = cross_val_xgb(params, X_train, y_train, folds, metric)

    print('\n\nStart Training on the whole dataset...')
    n_stop = np.int(mean_round * 1.1)
    final_clf = xgb.train(params, xgtrain, n_stop)

    return final_clf, metric, n_stop

###############################
# Create folds

cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=1337)
folds = list(cv.split(X,y))

# Quick validation to get a unique name
x_trn, x_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and validate
print("############ Final Classifier ######################")
clf, metric, n_stop = train(x_trn, x_val, y_trn, y_val, X, y, folds)

# Output
def output(X_test, sk_id_curr, classifier, n_stop, metric):
    print('Start predicting...')

    # XGBoost
    xgtest = xgb.DMatrix(X_test)
    predictions = classifier.predict(xgtest, ntree_limit=n_stop)

    #debug
    print('\n\nPredictions done. Here is a snippet')
    print(predictions)

    result = pd.DataFrame({
        'SK_ID_CURR': sk_id_curr,
        'TARGET': predictions
        })

    result.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str(metric)+'.csv', index=False)

df_test = pd.read_csv(open("./inputs/application_test.csv", "r"))
for feat, encoder in zip(categoricals, encoders):
    df_test[feat] = encoder.transform(df_test[feat].astype('str'))

output(df_test, df_test['SK_ID_CURR'], clf, n_stop, metric)
