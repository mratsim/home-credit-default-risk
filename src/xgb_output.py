# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import xgboost as xgb
import pandas as pd
import time
import logging
from src.xgb_features_importance import xgb_features_importance

## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

def xgb_output(X_test, sk_id_curr, classifier, n_stop, metric):
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

  str_metric = '{:.4f}'.format(round(metric,4))

  result.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str_metric+'.csv', index=False)

  # Features importance
  fi_by_importance, fi_by_feature = xgb_features_importance(classifier, X_test.columns)
  fi_by_importance.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str_metric+'-feature_importance_by_importance.csv', index=False)
  fi_by_feature.to_csv('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str_metric+'-feature_importance_by_feature.csv', index=False)

  print('\n\nHere are the top 40 important features')
  print(fi_by_importance.head(40))
