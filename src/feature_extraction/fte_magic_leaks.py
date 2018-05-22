# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
from src.instrumentation import logspeed

# A collection of features that predict by proxy
# but are not supposed to have any predictive power

@logspeed
def fte_magic_ids_leak(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      avg(b.SK_ID_BUREAU) as avg_magic_id_bureau,
      avg(p.SK_ID_PREV) as avg_magic_id_prev
    from
      {table} app
    left join
      bureau b on app.SK_ID_CURR = b.SK_ID_CURR
    left join
      previous_application p on app.SK_ID_CURR = p.SK_ID_CURR
    group by
      app.SK_ID_CURR
    order by
      app.SK_ID_CURR ASC
    """

    df[['avg_magic_id_bureau',
        'avg_magic_id_prev'
        ]] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
