# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_region(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      REGION_POPULATION_RELATIVE,
      REGION_RATING_CLIENT,
      REGION_RATING_CLIENT_W_CITY,
      REG_CITY_NOT_LIVE_CITY,
      REG_CITY_NOT_WORK_CITY,
      REG_REGION_NOT_LIVE_REGION,
      REG_REGION_NOT_WORK_REGION
    from
      {table}
    ORDER BY
      SK_ID_CURR ASC
    """

    df[['REGION_POPULATION_RELATIVE',
        'REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY',
        'REG_CITY_NOT_LIVE_CITY',
        'REG_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_LIVE_REGION',
        'REG_REGION_NOT_WORK_REGION']] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
