# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
import numpy as np

def fte_organisation(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query_mapper = f"""
    select
      distinct ORGANIZATION_TYPE
    from
      {table}
    order by
      ORGANIZATION_TYPE ASC;
    """

    df_mapper = pd.read_sql_query(query_mapper, db_conn)

    # Replace labels by their ID.
    # Note that "Industry: type 10" comes before "Industry: type 2"
    dict_mapper = {label: index for (index, label) in df_mapper['ORGANIZATION_TYPE'].iteritems()}
    dict_mapper['XNA'] = np.nan

    query = f"""
    select
      ORGANIZATION_TYPE
    from
      {table}
    order by
      SK_ID_CURR ASC;
    """

    df['org_type'] = pd.read_sql_query(query, db_conn)['ORGANIZATION_TYPE'].map(dict_mapper)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
