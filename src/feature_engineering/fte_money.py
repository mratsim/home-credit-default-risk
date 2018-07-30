# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
from src.instrumentation import logspeed

@logspeed
def fte_income_ratios(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      AMT_CREDIT / AMT_INCOME_TOTAL AS credit_income_ratio,
      AMT_ANNUITY / AMT_INCOME_TOTAL AS annuity_income_ratio,
      AMT_CREDIT / AMT_ANNUITY AS credit_annuity_ratio,
      AMT_CREDIT / AMT_GOODS_PRICE as credit_price_ratio,
      AMT_INCOME_TOTAL / (1e-5 + CNT_CHILDREN) AS income_per_child, --If no children, div by epsilon, ~infinite income
      AMT_INCOME_TOTAL / CNT_FAM_MEMBERS AS income_per_person,
      AMT_INCOME_TOTAL / (CNT_FAM_MEMBERS - CNT_CHILDREN) AS income_per_adult,
      AMT_CREDIT / (1e-5 + CNT_CHILDREN) AS credit_per_child, --If no children, div by epsilon, ~infinite income
      AMT_CREDIT / CNT_FAM_MEMBERS AS credit_per_person,
      AMT_CREDIT / (CNT_FAM_MEMBERS - CNT_CHILDREN) AS credit_per_adult,
      AMT_CREDIT,
      AMT_INCOME_TOTAL,
      AMT_ANNUITY
    from
      {table}
    order by
      SK_ID_CURR ASC;
    """

    df[[
      'credit_income_ratio',
      'annuity_income_ratio',
      'credit_annuity_ratio',
      'credit_price_ratio',
      'income_per_child',
      'income_per_person',
      'income_per_adult',
      'credit_per_child',
      'credit_per_person',
      'credit_per_adult',
      'AMT_CREDIT',
      'AMT_INCOME_TOTAL',
      'AMT_ANNUITY']] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file

def fte_goods_price(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      AMT_GOODS_PRICE,
      AMT_CREDIT / AMT_GOODS_PRICE AS credit_price_ratio
    from
      {table}
    order by
      SK_ID_CURR ASC;
    """

    df[['AMT_GOODS_PRICE', 'credit_price_ratio']] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
