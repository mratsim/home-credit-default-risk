# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_ext_source(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
    from
      {table}
    order by
      SK_ID_CURR ASC;
    """

    df[['EXT_SOURCE_1','EXT_SOURCE_2', 'EXT_SOURCE_3']] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
