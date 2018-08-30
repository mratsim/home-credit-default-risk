# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import logging
import pandas as pd
from src.instrumentation import logspeed
from src.cache import load_from_cache, save_to_cache

## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

@logspeed
def fte_missed_installments(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_missed_installments_train'
  cache_key_test = 'fte_missed_installments_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_missed_installments - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_missed_installments - Cache not found, will recompute from scratch')

  ########################################################

  query = """
  select
    SK_ID_CURR,
    AMT_INSTALMENT - AMT_PAYMENT AS DIFF_EXPECTED_PMT,
    DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT AS DAYS_LATE
  from
    installments_payments
  """

  installments_diff = pd.read_sql_query(query, db_conn)
  agg_installments_diff = installments_diff.groupby('SK_ID_CURR').agg(
    ["sum","mean","max","min","std", "count"]
  )
  agg_installments_diff.columns = pd.Index([e[0] +"_"+ e[1] for e in agg_installments_diff.columns.tolist()])

  train = train.merge(agg_installments_diff, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)
  test = test.merge(agg_installments_diff, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)

  ########################################################

  logger.info(f'Caching features in {cache_file}')
  train_cache = train[agg_installments_diff.columns]
  test_cache = test[agg_installments_diff.columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file
