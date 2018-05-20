# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
import numpy as np
from src.categorical_encoder import encode_categoricals

def fte_organisation(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    df['occupation_type']  = encode_categoricals(df, db_conn, table, 'OCCUPATION_TYPE')
    df['organisation_type']  = encode_categoricals(df, db_conn, table, 'ORGANIZATION_TYPE')

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file


def fte_work_phone(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      FLAG_WORK_PHONE
    from
      {table}
    order by
      SK_ID_CURR ASC;
    """

    df['FLAG_WORK_PHONE'] = pd.read_sql_query(query, db_conn)

  return train, test, y, db_conn, folds, cache_file
