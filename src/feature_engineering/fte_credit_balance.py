# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import logging
import pandas as pd
from src.instrumentation import logspeed
from src.cache import load_from_cache, save_to_cache

## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

@logspeed
def fte_withdrawals(train, test, y, db_conn, folds, cache_file):

  cache_key_train = 'fte_withdrawals_train'
  cache_key_test = 'fte_withdrawals_test'

  # Check if cache file exist and if data for this step is cached
  train_cached, test_cached = load_from_cache(cache_file, cache_key_train, cache_key_test)
  if train_cached is not None and test_cached is not None:
      logger.info('fte_withdrawals - Cache found, will use cached data')
      train = pd.concat([train, train_cached], axis = 1, copy = False)
      test = pd.concat([test, test_cached], axis = 1, copy = False)
      return train, test, y, db_conn, folds, cache_file

  logger.info('fte_withdrawals - Cache not found, will recompute from scratch')

  ########################################################

  def _trans(df, table, columns):
    query = f"""
    with instalments_per_app AS (
      select
        SK_ID_CURR,
        max(CNT_INSTALMENT_MATURE_CUM) AS nb_installments
      from
        credit_card_balance
      GROUP BY
        SK_ID_CURR, SK_ID_PREV
      ORDER BY
        SK_ID_CURR
      )
    select
      count(distinct SK_ID_PREV) as cb_nb_of_credit_cards,
      avg(CNT_DRAWINGS_ATM_CURRENT) as cb_avg_atm_withdrawal_count,
      avg(AMT_DRAWINGS_ATM_CURRENT) as cb_avg_atm_withdrawal_amount,
      sum(AMT_DRAWINGS_ATM_CURRENT) as cb_sum_atm_withdrawal_amount,
      avg(CNT_DRAWINGS_CURRENT) as cb_avg_withdrawal_count,
      avg(AMT_DRAWINGS_CURRENT) as cb_avg_withdrawal_amount,
      avg(CNT_DRAWINGS_POS_CURRENT) as cb_avg_pos_withdrawal_count,
      avg(AMT_DRAWINGS_POS_CURRENT) as cb_avg_pos_withdrawal_amount,
      avg(SK_DPD) as cb_avg_day_past_due,
      avg(SK_DPD_DEF) as cb_avg_day_past_due_tolerated
      --sum(nb_installments) as instalments_per_app
    FROM
      {table} app
    left join
      credit_card_balance ccb
        on app.SK_ID_CURR = ccb.SK_ID_CURR
    left join
      instalments_per_app ipa
        on ipa.SK_ID_CURR = ccb.SK_ID_CURR
    --where
    --  MONTHS_BALANCE BETWEEN -3 and -1
    group by
      ccb.sk_id_curr
    ORDER by
      app.SK_ID_CURR ASC
    """

    df[columns] = pd.read_sql_query(query, db_conn)

  columns = [
      'cb_nb_of_credit_cards',
      'cb_avg_atm_withdrawal_count',
      'cb_avg_atm_withdrawal_amount',
      'cb_sum_atm_withdrawal_amount',
      'cb_avg_withdrawal_count',
      'cb_avg_withdrawal_amount',
      'cb_avg_pos_withdrawal_count',
      'cb_avg_pos_withdrawal_amount',
      'cb_avg_day_past_due',
      'cb_avg_day_past_due_tolerated'
      # 'instalments_per_app'
      ]

  _trans(train, "application_train", columns)
  _trans(test, "application_test", columns)

  ########################################################

  logger.info(f'Caching features in {cache_file}')
  train_cache = train[columns]
  test_cache = test[columns]
  save_to_cache(cache_file, cache_key_train, cache_key_test, train_cache, test_cache)

  return train, test, y, db_conn, folds, cache_file
