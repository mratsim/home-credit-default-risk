# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import logging
import pandas as pd
from src.encoders import encode_categoricals
from src.instrumentation import logspeed
from src.cache import load_from_cache, save_to_cache

## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

@logspeed
def fte_application(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_application_train'
  cache_key_test = 'fte_application_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_application - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_application - Cache not found, will recompute from scratch')

  ########################################################

  def _trans(df, table, columns):
    query = f"""
    select
      OBS_30_CNT_SOCIAL_CIRCLE,
      DEF_30_CNT_SOCIAL_CIRCLE,
      OBS_60_CNT_SOCIAL_CIRCLE,
      DEF_60_CNT_SOCIAL_CIRCLE,
      REGION_POPULATION_RELATIVE,
      --REGION_RATING_CLIENT,
      REGION_RATING_CLIENT_W_CITY,
      REG_CITY_NOT_LIVE_CITY,
      --REG_CITY_NOT_WORK_CITY,
      --REG_REGION_NOT_LIVE_REGION
      --REG_REGION_NOT_WORK_REGION,
      --LIVE_REGION_NOT_WORK_REGION,
      CNT_CHILDREN,
      CNT_FAM_MEMBERS,
      case(CODE_GENDER) when "F" THEN 1 else 0 end isWoman,
      -- case(FLAG_OWN_CAR) when "Y" THEN 1 else 0 end ownCar,
      case(FLAG_OWN_REALTY) when "Y" THEN 1 else 0 end ownRealEstate,
      FLAG_DOCUMENT_3,
      FLOORSMAX_AVG,
      FLOORSMIN_AVG,
      YEARS_BUILD_AVG,
      LIVINGAREA_MODE,
      YEARS_BEGINEXPLUATATION_MODE,
      LANDAREA_MODE,
      BASEMENTAREA_MODE,
      TOTALAREA_MODE,
      APARTMENTS_AVG,
      COMMONAREA_AVG,
      NONLIVINGAREA_MEDI,
      --LIVINGAPARTMENTS_MODE,
      --YEARS_BUILD_MODE,
      --ENTRANCES_AVG,
      --AMT_REQ_CREDIT_BUREAU_DAY,
      --AMT_REQ_CREDIT_BUREAU_HOUR,
      AMT_REQ_CREDIT_BUREAU_MON,
      AMT_REQ_CREDIT_BUREAU_QRT,
      --AMT_REQ_CREDIT_BUREAU_WEEK,
      AMT_REQ_CREDIT_BUREAU_YEAR,
      FLAG_MOBIL,
      --FLAG_EMP_PHONE,
      FLAG_WORK_PHONE,
      --FLAG_CONT_MOBILE,
      --FLAG_PHONE,
      --FLAG_EMAIL,
      EXT_SOURCE_1,
      EXT_SOURCE_2,
      EXT_SOURCE_3
    from
      {table}
    ORDER BY
      SK_ID_CURR ASC
    """

    df[columns] = pd.read_sql_query(query, db_conn)

  columns = [
      'OBS_30_CNT_SOCIAL_CIRCLE',
      'DEF_30_CNT_SOCIAL_CIRCLE',
      'OBS_60_CNT_SOCIAL_CIRCLE',
      'DEF_60_CNT_SOCIAL_CIRCLE',
      'REGION_POPULATION_RELATIVE',
      # 'REGION_RATING_CLIENT',
      'REGION_RATING_CLIENT_W_CITY',
      'REG_CITY_NOT_LIVE_CITY',
      # 'REG_CITY_NOT_WORK_CITY',
      # 'REG_REGION_NOT_LIVE_REGION',
      # 'REG_REGION_NOT_WORK_REGION',
      # 'LIVE_REGION_NOT_WORK_REGION',
      'CNT_CHILDREN',
      'CNT_FAM_MEMBERS',
      'isWoman',
      # 'FLAG_OWN_CAR',
      'FLAG_OWN_REALTY',
      'FLAG_DOCUMENT_3',
      'FLOORSMAX_AVG',
      'FLOORSMIN_AVG',
      'YEARS_BUILD_AVG',
      'LIVINGAREA_MODE',
      'YEARS_BEGINEXPLUATATION_MODE',
      'LANDAREA_MODE',
      'BASEMENTAREA_MODE',
      'TOTALAREA_MODE',
      'APARTMENTS_AVG',
      'COMMONAREA_AVG',
      'NONLIVINGAREA_MEDI',
      # 'LIVINGAPARTMENTS_MODE',
      # 'YEARS_BUILD_MODE',
      # 'ENTRANCES_AVG',
      # 'AMT_REQ_CREDIT_BUREAU_DAY',
      # 'AMT_REQ_CREDIT_BUREAU_HOUR',
      'AMT_REQ_CREDIT_BUREAU_MON',
      'AMT_REQ_CREDIT_BUREAU_QRT',
      # 'AMT_REQ_CREDIT_BUREAU_WEEK',
      'AMT_REQ_CREDIT_BUREAU_YEAR',
      'FLAG_MOBIL',
      #'FLAG_EMP_PHONE',
      'FLAG_WORK_PHONE',
      #'FLAG_CONT_MOBILE',
      #'FLAG_PHONE',
      #'FLAG_EMAIL',
      'EXT_SOURCE_1',
      'EXT_SOURCE_2',
      'EXT_SOURCE_3'
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
def fte_app_categoricals(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_app_categoricals_train'
  cache_key_test = 'fte_app_categoricals_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_app_categoricals - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_app_categoricals - Cache not found, will recompute from scratch')

  ########################################################

  def _trans(df, table):
    df['NAME_CONTRACT_TYPE']  = encode_categoricals(df, db_conn, table, 'NAME_CONTRACT_TYPE')
    df['NAME_TYPE_SUITE'] = encode_categoricals(df, db_conn, table, 'NAME_TYPE_SUITE')
    df['OCCUPATION_TYPE']  = encode_categoricals(df, db_conn, table, 'OCCUPATION_TYPE')
    df['ORGANIZATION_TYPE']  = encode_categoricals(df, db_conn, table, 'ORGANIZATION_TYPE')
    df['NAME_INCOME_TYPE']    = encode_categoricals(df, db_conn, table, 'NAME_INCOME_TYPE')
    df['NAME_EDUCATION_TYPE'] = encode_categoricals(df, db_conn, table, 'NAME_EDUCATION_TYPE')
    df['NAME_FAMILY_STATUS']  = encode_categoricals(df, db_conn, table, 'NAME_FAMILY_STATUS')
    df['NAME_HOUSING_TYPE']  = encode_categoricals(df, db_conn, table, 'NAME_HOUSING_TYPE')

  _trans(train, "application_train")
  _trans(test, "application_test")

  ########################################################

  columns =  ['NAME_CONTRACT_TYPE',
              'NAME_TYPE_SUITE',
              'OCCUPATION_TYPE',
              'ORGANIZATION_TYPE',
              'NAME_INCOME_TYPE',
              'NAME_EDUCATION_TYPE',
              'NAME_FAMILY_STATUS',
              'NAME_HOUSING_TYPE']
  logger.info(f'Caching features in {cache_file}')
  train_cache = train[columns]
  test_cache = test[columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file
