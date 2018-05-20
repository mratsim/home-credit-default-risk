# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_withdrawals(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      avg(CNT_DRAWINGS_ATM_CURRENT) as cb_avg_atm_withdrawal_count,
      avg(AMT_DRAWINGS_ATM_CURRENT) as cb_avg_atm_withdrawal_amount,
      avg(CNT_DRAWINGS_CURRENT) as cb_avg_withdrawal_count,
      avg(AMT_DRAWINGS_CURRENT) as cb_avg_withdrawal_amount,
      avg(CNT_DRAWINGS_POS_CURRENT) as cb_avg_pos_withdrawal_count,
      avg(AMT_DRAWINGS_POS_CURRENT) as cb_avg_pos_withdrawal_amount,
      avg(SK_DPD) as cb_avg_day_past_due,
      avg(SK_DPD_DEF) as cb_avg_day_past_due_tolerated
    FROM
      {table} app
    left join
      credit_card_balance ccb
        on app.SK_ID_CURR = ccb.SK_ID_CURR
    -- where
    --  MONTHS_BALANCE BETWEEN -3 and -1
    group by
      ccb.sk_id_curr
    ORDER by
      app.SK_ID_CURR ASC
    """

    df[[
      'cb_avg_atm_withdrawal_count',
      'cb_avg_atm_withdrawal_amount',
      'cb_avg_withdrawal_count',
      'cb_avg_withdrawal_amount',
      'cb_avg_pos_withdrawal_count',
      'cb_avg_pos_withdrawal_amount',
      'cb_avg_day_past_due',
      'cb_avg_day_past_due_tolerated'
      ]] = pd.read_sql_query(query, db_conn)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
