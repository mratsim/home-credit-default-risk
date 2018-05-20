# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import xgboost as xgb
import pandas as pd
import time
from src.xgb_features_importance import xgb_features_importance

def xgb_output(X_test, sk_id_curr, classifier, n_stop, metric):
  print('Start predicting...')

  # XGBoost
  xgtest = xgb.DMatrix(X_test)
  predictions = classifier.predict(xgtest, ntree_limit=n_stop)

  # debug
  print('\n\nPredictions done. Here is a snippet')
  print(predictions)

  result = pd.DataFrame({
    'SK_ID_CURR': sk_id_curr,
    'TARGET': predictions
    })

  result.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str(metric)+'.csv', index=False)

  # Features importance
  features_importance = xgb_features_importance(classifier, X_test.columns)
  features_importance.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str(metric)+'-feature_importance.csv', index=False)

  print('\n\nHere are the top 20 important features')
  print(features_importance.head(20))
