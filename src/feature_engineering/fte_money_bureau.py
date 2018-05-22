# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.
import logging
import pandas as pd
from src.instrumentation import logspeed
from src.cache import load_from_cache, save_to_cache


## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

@logspeed
def fte_bureau_credit_situation(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_bureau_credit_situation_train'
  cache_key_test = 'fte_bureau_credit_situation_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_bureau_credit_situation - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_bureau_credit_situation - Cache not found, will recompute from scratch')

  ########################################################

  def _trans(df, table, columns):
    query = f"""
    select
      IFNULL(count(b.SK_ID_BUREAU), 0) AS b_total_prev_applications,
      IFNULL(sum(case CREDIT_ACTIVE when 'Active' then 1 else 0 end), 0) AS b_current_active_applications,
      IFNULL(sum(AMT_CREDIT_SUM), 0) AS b_total_prev_credit,
      IFNULL(sum(case CREDIT_ACTIVE when 'Active' then AMT_CREDIT_SUM else 0 end), 0) AS b_active_credit_amount,
      IFNULL(sum(AMT_CREDIT_SUM_DEBT), 0) AS b_current_debt,
      IFNULL(sum(AMT_CREDIT_MAX_OVERDUE), 0) AS b_current_overdue,
      IFNULL(max(-DAYS_CREDIT), 99 * 365.25) / 365.25 AS b_first_credit_years_ago,
      IFNULL(min(-DAYS_CREDIT), 99 * 365.25) / 365.25 AS b_last_credit_years_ago,
      IFNULL(max(DAYS_CREDIT_ENDDATE), -99 * 365.25) / 365.25 AS b_existing_credit_close_date,
      IFNULL(max(-DAYS_ENDDATE_FACT), 99 * 365.25) / 365.25 AS b_years_since_no_card_credit
      -- IFNULL(min(-DAYS_CREDIT_UPDATE), 99 * 365.25) AS b_last_DAYS_CREDIT_UPDATE
    from
      {table} app
    left join
      bureau b on app.SK_ID_CURR = b.SK_ID_CURR
    GROUP BY
      app.SK_ID_CURR
    ORDER BY
      app.SK_ID_CURR ASC;
    """

    df[columns] = pd.read_sql_query(query, db_conn)

    # TODO add currency, otherwise credit is not comparable

  columns = ['b_total_prev_applications',
        'b_current_active_applications',
        'b_total_prev_credit',
        'b_active_credit_amount',
        'b_current_debt',
        'b_current_overdue',
        'b_first_credit_years_ago',
        'b_last_credit_years_ago',
        'b_existing_credit_close_date',
        'b_years_since_no_card_credit'
        # 'b_last_DAYS_CREDIT_UPDATE'
        ]

  _trans(train, "application_train", columns)
  _trans(test, "application_test", columns)

  ########################################################

  logger.info(f'Caching features in {cache_file}')
  train_cache = train[columns]
  test_cache = test[columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file
