# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_age(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      -DAYS_BIRTH / 365.25 AS years_birth,
      -DAYS_EMPLOYED / 365.25 AS years_employed,
      -DAYS_REGISTRATION / 365.25 AS years_registration,
      -DAYS_ID_PUBLISH / 365.25 AS years_id_publish,
      case DAYS_LAST_PHONE_CHANGE
        when 0 then -999
        else -DAYS_LAST_PHONE_CHANGE / 365.25
      end years_last_phone_change
    from
      {table};
    """

    df[['years_birth', 'years_employed', 'years_registration', 'years_id_publish', 'years_last_phone_change']] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
