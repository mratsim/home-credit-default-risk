# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_building(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      FLOORSMAX_AVG,
      FLOORSMIN_AVG,
      LIVINGAREA_MODE,
      YEARS_BEGINEXPLUATATION_MODE,
      LANDAREA_MODE,
      BASEMENTAREA_MODE,
      TOTALAREA_MODE,
      APARTMENTS_AVG,
      COMMONAREA_AVG,
      NONLIVINGAREA_MEDI
      --LIVINGAPARTMENTS_MODE,
      --YEARS_BUILD_MODE,
      --ENTRANCES_AVG
    from
      {table}
    ORDER BY
      SK_ID_CURR ASC
    """

    df[['FLOORSMAX_AVG',
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
        # 'ENTRANCES_AVG'
        ]] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
