# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
from src.encoders import encode_categoricals
from src.instrumentation import logspeed

@logspeed
def fte_application(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
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
      case(FLAG_OWN_CAR) when "Y" THEN 1 else 0 end ownCar,
      case(FLAG_OWN_REALTY) when "Y" THEN 1 else 0 end ownRealEstate,
      FLAG_DOCUMENT_3,
      FLOORSMAX_AVG,
      FLOORSMIN_AVG,
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
      FLAG_WORK_PHONE,
      EXT_SOURCE_1,
      EXT_SOURCE_2,
      EXT_SOURCE_3
    from
      {table}
    ORDER BY
      SK_ID_CURR ASC
    """

    df[[
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
        'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY',
        'FLAG_DOCUMENT_3',
        'FLOORSMAX_AVG',
        'FLOORSMIN_AVG',
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
        'FLAG_WORK_PHONE',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3'
        ]] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file

def fte_app_categoricals(train, test, y, db_conn, folds, cache_file):
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

  return train, test, y, db_conn, folds, cache_file
