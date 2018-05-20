# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
import numpy as np

def xgb_features_importance(classifier, feat_names):
  importance = classifier.get_fscore()
  result = pd.DataFrame(importance,index=np.arange(2)).T
  result.iloc[:,0]= result.index
  result.columns=['feature','importance']
  result.sort_values('importance', inplace=True, ascending=False)
  result.reset_index(drop=True, inplace=True)
  return result
