# Copyright 2017-2018 Mamy André-Ratsimbazafy. All rights reserved.

from functools import reduce
from itertools import starmap
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import pandas as pd
from scipy import sparse
import numpy as np


############ Helpers Functional Programming####################
#compose list of functions (chained composition)
def compose(*funcs):
  def _compose(f, g):
    # functions are expecting X,y not (X,y) so must unpack with *g
    return lambda *args, **kwargs: f(*g(*args, **kwargs))
  return reduce(_compose, funcs)

# zipWith
def zip_with(f, list_of_tuple):
  return starmap(f, zip(*list_of_tuple))

############# Helper for concatenation ###################
# Concat columns of dataframes
def _concat_col_pd(*list_of_df):
  return pd.concat([*list_of_df], axis=1)

def _convert_2D(feat):
  """
  Convert 1-dimensional arrays to 2-dimensional column vectors.
  """
  if len(feat.shape) == 1:
    feat = np.array([feat]).T
  return feat

def _concat_col(*list_of_arrays):
  list_of_arrays = list(map(_convert_2D, list_of_arrays))
  if any(map(sparse.issparse, list_of_arrays)):
    return sparse.hstack(list_of_arrays).tocsr()
  return np.hstack(list_of_arrays)

# pipe functions, reverse the order so that it's in the usual FIFO function order
def feat_engineering_pipe(*funcs):
  return compose(*reversed(funcs))

def feat_selection(ft_selection, df_train, df_val_test, y=None,out_type=None):
  trn = df_train.copy()
  valtst = df_val_test.copy()

  ft_selection = [(trn,valtst,label, y,transfo) for (label,transfo) in ft_selection]

  if out_type == 'dataframe':
    tuples_trn_val_test = starmap(_feat_transfo_df, ft_selection)
    trn, valtst = zip_with(_concat_col_pd, tuples_trn_val_test)

  else:
    tuples_trn_val_test = starmap(_feat_transfo, ft_selection)
    trn, valtst = zip_with(_concat_col, tuples_trn_val_test)

  return trn, valtst

def _list_to_pipe_transformer(transformers):
  if isinstance(transformers, list):
    transformers = make_pipeline(*transformers)
  return transformers

def _feat_transfo(train, val_test, sCol, y=None, Transformer=None):
  if Transformer is None:
    #make sure we return the same whether its "feature" or ["feature"]
    return (train[sCol].values, val_test[sCol].values)

  Transformer = _list_to_pipe_transformer(Transformer)

  trn = Transformer.fit_transform(train[sCol].values,y)
  valtst = Transformer.transform(val_test[sCol].values)

  return (trn, valtst)

def _feat_transfo_df(train,val_test, sCol, y=None, Transformer=None):
  if Transformer is None:
    return (train[sCol], val_test[sCol])

  Transformer = _list_to_pipe_transformer(Transformer)

  def _trans(df, sCol, y, Transformer,flag):
    if flag == "fit_transform":
      transformed = Transformer.fit_transform(df[sCol], y).T
    elif flag == "transform":
      transformed = Transformer.transform(df[sCol]).T
    #else Raise exception

    if isinstance(sCol, list):
      label = sCol[0]
    else:
      label = sCol

    feature_list = []
    n = 0
    # feat_label will hold the list of descriptive names
    feat_label = ''
    for serie in transformed:
      if isinstance(Transformer, OneHotEncoder):
        feat_label = label + '_' + str(Transformer.active_features_[n])
      else:
        try:
          feat_label = label + '_' + Transformer.classes_[n] # LabelBinarizer
        except:
          feat_label = label
      # To keep the index, we need to assign to the original DF and then extract again
      df[feat_label] = serie
      feature_list.append(feat_label)
    return df[feature_list]

  trn = _trans(train, y, sCol, Transformer,'fit_transform')
  valtst = _trans(val_test, y, sCol, Transformer,'transform')

  return (trn, valtst)

######################################
# Multiprocessing function
# from multiprocessing import Pool, cpu_count

# Note: make sure you work on copies and not on the original DF
# You may have race conditions and unexpected behaviour

## Multiprocessing is slower and obfuscate errors (MemoryError or ThreadLock instead of the real issue)
## This is probably due to false sharing / cache invalidation.
# with Pool(cpu_count()) as mp:
#     tuples_trn_val_test = mp.starmap(_feat_transfo,ft_selection)
#     trn, val, tst = par_zip_with(_concat_col,tuples_trn_val_test, mp)

# Parallel zipWith
# def par_zip_with(f, list_of_tuple, pool):
#     return pool.starmap(f, zip(*list_of_tuple))
