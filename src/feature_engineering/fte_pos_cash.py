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
  dict_train, dict_test = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if dict_train is not None and dict_test is not None:
      logger.info('Cache found: fte_pos_cash_aggregate will use cached data')
      train = train.assign(**dict_train)
      test = test.assign(**dict_test)
      return train, test, y, db_conn, folds, cache_file

  logger.info('Cache not found: fte_pos_cash_aggregate will recompute from scratch')

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
          SK_DPD = ["sum","mean","max","min","std"],
          SK_DPD_DEF = ["sum","mean","max","min","std"],
          SK_ID_CURR = 'count')
      )
  agg_POS_CASH.columns = pd.Index([e[0] +"_"+ e[1] for e in agg_POS_CASH.columns.tolist()])

  train = train.merge(agg_POS_CASH, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)
  test = test.merge(agg_POS_CASH, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)

  ########################################################

  logger.info(f'Caching features in {cache_file}')
  train_cache = train[agg_POS_CASH.columns].to_dict()
  test_cache = test[agg_POS_CASH.columns].to_dict()
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file

@logspeed
def fte_pos_cash_current_status(train, test, y, db_conn, folds, cache_file):
  ## Count the still active/completed/amortized debt credit

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

  train = train.merge(pos_cash_current, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)
  test = test.merge(pos_cash_current, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)

  return train, test, y, db_conn, folds, cache_file
