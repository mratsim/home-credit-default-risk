# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import logging
import pandas as pd
from src.encoders import encode_average
from src.instrumentation import logspeed
from src.cache import load_from_cache, save_to_cache


## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

@logspeed
def fte_prev_credit_situation(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_prev_credit_situation_train'
  cache_key_test = 'fte_prev_credit_situation_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_prev_credit_situation - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_prev_credit_situation - Cache not found, will recompute from scratch')

  ########################################################

  def _trans(df, table, columns):
    query = f"""
    select
      IFNULL(sum(case(p.NAME_CONTRACT_TYPE) when 'Consumer loans' then 1 else 0 end), 0) AS p_total_consumer_loans,
      IFNULL(sum(case(p.NAME_CONTRACT_TYPE) when 'Cash loans' then 1 else 0 end), 0) AS p_total_cash_loans,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Refused' then 1 else 0 end), 0) AS p_total_application_refused,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Canceled' then 1 else 0 end), 0) AS p_total_application_canceled,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Approved' then 1 else 0 end), 0) AS p_total_application_approved,
      -- IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Unused offer' then 1 else 0 end), 0) AS p_total_application_unused,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Refused' then p.AMT_APPLICATION else 0 end), 0) AS p_avg_application_refused,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Canceled' then p.AMT_APPLICATION else 0 end), 0) AS p_avg_application_canceled,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Approved' then p.AMT_APPLICATION else 0 end), 0) AS p_avg_application_approved,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Unused offer' then p.AMT_APPLICATION else 0 end), 0) AS p_avg_application_unused,
      avg(case(p.NAME_CONTRACT_STATUS) when 'Approved' then 1.0 when 'Unused offer' then 1.0 else 0.0 end) AS p_ratio_app_approved_total,
      IFNULL(avg(p.AMT_APPLICATION - p.AMT_CREDIT), 0) AS p_avg_diff_asked_offered,
      IFNULL(avg(p.CNT_PAYMENT), 0) AS p_avg_payment_schedule,
      IFNULL(avg(p.AMT_ANNUITY), 0) AS p_avg_annuity,
      IFNULL(avg(p.CNT_PAYMENT), 0) / app.AMT_ANNUITY AS p_ratio_annuity_p_app,
      avg(p.AMT_DOWN_PAYMENT) AS p_avg_down_payment
    from
      {table} app
    left join
      previous_application p on app.SK_ID_CURR = p.SK_ID_CURR
    GROUP BY
      app.SK_ID_CURR
    ORDER BY
      app.SK_ID_CURR ASC;
    """

    df[columns] = pd.read_sql_query(query, db_conn)

    # TODO add currency, otherwise credit is not comparable

  columns = ['p_total_consumer_loans',
        'p_total_cash_loans',
        'p_total_application_refused',
        'p_total_application_canceled',
        'p_total_application_approved',
        # 'p_total_application_unused',
        'p_avg_application_refused',
        'p_avg_application_canceled',
        'p_avg_application_approved',
        'p_avg_application_unused',
        'p_ratio_app_approved_total',
        'p_avg_diff_asked_offered',
        'p_avg_payment_schedule',
        'p_avg_annuity',
        'p_ratio_annuity_p_app',
        'p_avg_down_payment']

  _trans(train, "application_train", columns)
  _trans(test, "application_test", columns)

  ########################################################

  logger.info(f'Caching features in {cache_file}')
  train_cache = train[columns]
  test_cache = test[columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file

@logspeed
def fte_prev_app_process(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_prev_app_process_train'
  cache_key_test = 'fte_prev_app_process_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_prev_app_process - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_prev_app_process - Cache not found, will recompute from scratch')

  ########################################################

  def _trans(df, table, columns):
    query = f"""
    select
      avg(-p.DAYS_DECISION) / 365.25 p_avg_years_decision,
      avg(case(p.NAME_CONTRACT_TYPE)
        when 'Revolving loans' Then NULL
        else -p.DAYS_FIRST_DUE
        end
      ) / 365.25 p_avg_years_first_due,
      avg(case(p.NAME_CONTRACT_TYPE)
        when 'Revolving loans' Then NULL
        else -p.DAYS_LAST_DUE_1ST_VERSION
        end
      ) / 365.25 p_avg_years_last_due_1st_version,
      avg(p.HOUR_APPR_PROCESS_START) p_avg_hour_start,
      sum(case(p.NAME_CONTRACT_TYPE)
        when 'Revolving loans' Then 1
        else 0
        end
      ) p_count_revolving_loans
    from
      {table} app
    left join
      previous_application p on app.SK_ID_CURR = p.SK_ID_CURR
    GROUP BY
      app.SK_ID_CURR
    ORDER BY
      app.SK_ID_CURR ASC;
    """

    df[columns] = pd.read_sql_query(query, db_conn)

    # TODO add currency, otherwise credit is not comparable

  columns = ['p_avg_years_decision',
        'p_avg_years_first_due',
        'p_avg_years_last_due_1st_version',
        'p_avg_hour_start',
        'p_count_revolving_loans'
        ]

  _trans(train, "application_train", columns)
  _trans(test, "application_test", columns)

  ########################################################

  logger.info(f'Caching features in {cache_file}')
  train_cache = train[columns]
  test_cache = test[columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file

@logspeed
def fte_sales_channels(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_sales_channels_train'
  cache_key_test = 'fte_sales_channels_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_sales_channels - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_sales_channels - Cache not found, will recompute from scratch')

  ########################################################

  def _trans(df, table):
    df['p_avg_seller_size'] = encode_average(df, db_conn, table, 'previous_application', 'SELLERPLACE_AREA')
    df['p_avg_channel_size'] = encode_average(df, db_conn, table, 'previous_application', 'CHANNEL_TYPE')

  _trans(train, "application_train")
  _trans(test, "application_test")

  ########################################################

  columns = ['p_avg_seller_size', 'p_avg_channel_size']
  logger.info(f'Caching features in {cache_file}')
  train_cache = train[columns]
  test_cache = test[columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file
