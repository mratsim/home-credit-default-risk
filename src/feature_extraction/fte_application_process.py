# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_application_process(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      FLAG_DOCUMENT_3
    from
      {table}
    ORDER BY
      SK_ID_CURR ASC;
    """

    df['FLAG_DOCUMENT_3'] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
