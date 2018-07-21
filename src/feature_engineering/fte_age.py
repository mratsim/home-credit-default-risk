# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
from src.instrumentation import logspeed

@logspeed
def fte_age(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    with raw_age_query AS (
      select
        SK_ID_CURR,
        -DAYS_BIRTH / 365.25 AS years_birth,
        -DAYS_EMPLOYED / 365.25 AS years_employed,
        -DAYS_REGISTRATION / 365.25 AS years_registration,
        -DAYS_ID_PUBLISH / 365.25 AS years_id_publish,
        case DAYS_LAST_PHONE_CHANGE
          when 0 then NULL
          else -DAYS_LAST_PHONE_CHANGE / 365.25
        end years_last_phone_change,
        OWN_CAR_AGE,
        CNT_FAM_MEMBERS - CNT_CHILDREN AS count_adults
      from
        {table}
      order by
        SK_ID_CURR ASC
      )
    select
      years_birth,
      years_employed,
      years_registration,
      years_id_publish,
      years_last_phone_change,
      OWN_CAR_AGE,
      count_adults,
      OWN_CAR_AGE / years_birth AS car_age_birth_ratio,
      OWN_CAR_AGE / years_employed AS car_age_employment_ratio,
      years_employed / years_birth AS employment_birth_ratio,
      years_last_phone_change / years_birth AS phone_age_birth_ratio,
      years_last_phone_change / years_employed AS phone_age_employment_ratio
    from
      raw_age_query
    order by
      SK_ID_CURR ASC
    """

    df[[
      'years_birth',
      'years_employed',
      'years_registration',
      'years_id_publish',
      'years_last_phone_change',
      'OWN_CAR_AGE',
      'count_adults',
      'car_age_birth_ratio',
      'car_age_employment_ratio',
      'employment_birth_ratio',
      'phone_age_birth_ratio',
      'phone_age_employment_ratio'
      ]] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
