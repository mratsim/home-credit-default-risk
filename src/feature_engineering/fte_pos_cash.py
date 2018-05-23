# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
import logging
from src.instrumentation import logspeed
from src.cache import load_from_cache, save_to_cache

## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

@logspeed
def fte_pos_cash_aggregate(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_pos_cash_aggregate_train'
  cache_key_test = 'fte_pos_cash_aggregate_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_pos_cash_aggregate - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_pos_cash_aggregate - Cache not found, will recompute from scratch')

  ########################################################

  # SQLite doesn't have stddev function, revert to Pandas
  # Note that the DB has 10M rows
  pos_cash = pd.read_sql_query(
    'select * FROM POS_CASH_balance ORDER BY SK_ID_CURR ASC;',
    db_conn
    )

  # Create the aggregate
  agg_POS_CASH = pos_cash.groupby('SK_ID_CURR').agg(
      dict(MONTHS_BALANCE = ["sum","mean","max","min","std"],
          CNT_INSTALMENT = ["sum","mean","max","min","std"],
          CNT_INSTALMENT_FUTURE = ["sum","mean","max","min","std"],
          SK_DPD = ["sum","mean","max","std"], # dropping mean
          SK_DPD_DEF = ["sum","mean","max","min","std"],
          SK_ID_CURR = 'count')
      )
  agg_POS_CASH.columns = pd.Index([e[0] +"_"+ e[1] for e in agg_POS_CASH.columns.tolist()])

  train = train.merge(agg_POS_CASH, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)
  test = test.merge(agg_POS_CASH, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)

  ########################################################

  logger.info(f'Caching features in {cache_file}')
  train_cache = train[agg_POS_CASH.columns]
  test_cache = test[agg_POS_CASH.columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file

@logspeed
def fte_pos_cash_current_status(train, test, y, db_conn, folds, cache_file):
  ## Count the still active/completed/amortized debt credit

  cache_key_train = 'fte_pos_cash_current_status_train'
  cache_key_test = 'fte_pos_cash_current_status_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_pos_cash_current_status - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_pos_cash_current_status - Cache not found, will recompute from scratch')

  ########################################################

  # In SQLite we avoid joining on temporary tables/subqueries/with-statement as they are not indexed
  # and super-slow. ORDER BY on the following result is very slow to ...
  # (~>1min vs 20ms for just dumping the result on i5-5257U)

  query = """
  SELECT
    SK_ID_CURR, -- SK_ID_PREV,
    NAME_CONTRACT_STATUS
  FROM
    POS_CASH_balance
  GROUP BY
    SK_ID_PREV
	HAVING
	  MONTHS_BALANCE = max(MONTHS_BALANCE)
  """

  pos_cash_current = pd.read_sql_query(query, db_conn)
  # Pivot
  pos_cash_current = pd.get_dummies(pos_cash_current, columns=['NAME_CONTRACT_STATUS']).groupby('SK_ID_CURR').sum()

  # TODO: add a proper feature selection phase
  pos_cash_current.drop(columns=['NAME_CONTRACT_STATUS_Signed', 'NAME_CONTRACT_STATUS_Returned to the store'])

  train = train.merge(pos_cash_current, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)
  test = test.merge(pos_cash_current, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)

  ########################################################

  logger.info(f'Caching features in {cache_file}')
  train_cache = train[pos_cash_current.columns]
  test_cache = test[pos_cash_current.columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file
