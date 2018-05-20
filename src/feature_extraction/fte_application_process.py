# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
from src.encoders import encode_categoricals

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

def fte_contract(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    df['NAME_CONTRACT_TYPE']  = encode_categoricals(df, db_conn, table, 'NAME_CONTRACT_TYPE')
    df['NAME_TYPE_SUITE'] = encode_categoricals(df, db_conn, table, 'NAME_TYPE_SUITE')

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
