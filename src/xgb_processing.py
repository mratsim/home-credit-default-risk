# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import xgboost as xgb
import pandas as pd
import numpy as np
import time
import logging
from sklearn.metrics import roc_auc_score
from src.xgb_features_importance import xgb_features_importance

## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

def xgb_validate(x_trn, x_val, y_trn, y_val, xgb_params, seed_val=0, num_rounds=4096):

  num_rounds = num_rounds
  xgtrain = xgb.DMatrix(x_trn, label=y_trn)
  xgtest = xgb.DMatrix(x_val, label=y_val)

  logger.info('Start Validation training on 80% of the dataset...')

  # train
  watchlist = [ (xgtest, 'test') ]
  model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
  logger.info('End trainind on 80% of the dataset...')
  logger.info('Start validating prediction on 20% unseen data')

  # predict
  y_pred = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

  # eval
  val_score = roc_auc_score(y_val, y_pred)
  logger.info(f'We stopped at boosting round:  {model.best_ntree_limit}')
  logger.info(f'The ROC AUC on validation set is: {val_score}\n')

  return val_score

# Ideally cross val split should be done before feature engineering,
# and feature engineering + selection should be done separately for each splits
# so it better mimics out-of-sample predictions
def xgb_cross_val(params, X, y, folds):

  logger.info('Cross validating Stratified 7-fold... and retrieving best stopping round.\n')
  n = 1
  num_rounds = 3000

  list_rounds = []
  list_scores = []

  for train_idx, valid_idx in folds:
    logger.info('#################################')
    logger.info(f'#########  Validating for fold: {n}')

    xgtrain = xgb.DMatrix(X.values[train_idx], label=y.values[train_idx])
    xgtest = xgb.DMatrix(X.values[valid_idx], label=y.values[valid_idx])

    watchlist = [ (xgtest, 'test') ]
    model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=False)

    rounds = model.best_ntree_limit
    score = model.best_score

    logger.info(f'Fold {n} - best round: {rounds}')
    logger.info(f'Fold {n} - best score: {score}\n')

    list_rounds.append(rounds)
    list_scores.append(score)
    n += 1

  mean_round = np.mean(list_rounds)

  logger.info(f'End cross validating {n-1} folds') #otherwise it displays 6 folds
  logger.info("Cross Validation Scores are: " + str(np.round(list_scores,4)))
  logger.info(f'Mean CrossVal score is:  {np.mean(list_scores):.4f}')
  logger.info(f'Std Dev CrossVal score is: {np.std(list_scores):.4f}')
  logger.info("Cross Validation early stopping rounds are: " + str(np.round(list_rounds,4)))
  logger.info(f'Mean early stopping round is: {mean_round:.4f}')
  logger.info(f'Std Dev early stopping round is: {np.std(list_rounds):.4f}\n')

  return mean_round

def xgb_features_importance(classifier, feat_names):
  importance = classifier.get_fscore()
  result = pd.DataFrame(importance,index=np.arange(2)).T
  result.iloc[:,0]= result.index
  result.columns=['feature','importance']

  result_by_importance = result.sort_values('importance', inplace=False, ascending=False)
  result_by_importance.reset_index(drop=True, inplace=True)

  result_by_feature = result.sort_values('feature', inplace=False, ascending=True)
  result_by_feature.reset_index(drop=True, inplace=True)

  return result_by_importance, result_by_feature

def xgb_output(X_test, sk_id_curr, classifier, n_stop, val_score):
  # XGBoost
  xgtest = xgb.DMatrix(X_test)
  predictions = classifier.predict(xgtest, ntree_limit=n_stop)

  # debug
  logger.info('\n\nPredictions done. Here is a snippet')
  logger.info(predictions)

  result = pd.DataFrame({
  'SK_ID_CURR': sk_id_curr,
  'TARGET': predictions
  })

  str_val_score = '{:.4f}'.format(round(val_score,4))

  result.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str_val_score+'.csv', index=False)

  # Features importance
  fi_by_importance, fi_by_feature = xgb_features_importance(classifier, X_test.columns)
  fi_by_importance.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str_val_score+'-feature_importance_by_importance.csv', index=False)
  fi_by_feature.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str_val_score+'-feature_importance_by_feature.csv', index=False)

  print('\n\nHere are the top 40 important features')
  print(fi_by_importance.head(40))
