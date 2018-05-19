# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_income_ratios(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      AMT_CREDIT / AMT_INCOME_TOTAL AS credit_income_ratio,
      AMT_ANNUITY / AMT_INCOME_TOTAL AS annuity_income_ratio
    from
      {table};
    """

    df[['credit_income_ratio', 'annuity_income_ratio']] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
