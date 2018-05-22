# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
from src.instrumentation import logspeed

@logspeed
def fte_pos_cash_aggregate(train, test, y, db_conn, folds, cache_file):

  # SQLite doesn't have stddev function, revert to Pandas
  # Note that the DB has 10M rows
  pos_cash = pd.read_sql_query(
    'select * FROM POS_CASH_balance ORDER BY SK_ID_CURR ASC;',
    db_conn
    )

  # Create the aggregate
  agg_POS_CASH = pos_cash.groupby('SK_ID_CURR').agg(
      dict(MONTHS_BALANCE = ["sum","mean","max","min","std"],
          CNT_INSTALMENT = ["sum","mean","max","min","std"],
          CNT_INSTALMENT_FUTURE = ["sum","mean","max","min","std"],
          SK_DPD = ["sum","mean","max","min","std"],
          SK_DPD_DEF = ["sum","mean","max","min","std"],
          SK_ID_CURR = 'count')
      )
  agg_POS_CASH.columns = pd.Index([e[0] +"_"+ e[1] for e in agg_POS_CASH.columns.tolist()])

  train = train.merge(agg_POS_CASH, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)
  test = test.merge(agg_POS_CASH, left_on='SK_ID_CURR', right_index=True, how = 'left', copy = False)

  ## Count the still active credits
  ## This is slow on SQLite (single threaded + no index for the subquery/view) ~55s on my i5-5227U dual-core
  #
  # WITH current_poscash as (
  # 	select
  # 	  SK_ID_CURR, -- SK_ID_PREV,
  # 	  case when
  # 	    (MONTHS_BALANCE = max(MONTHS_BALANCE))
  # 	    and
  # 	    (NAME_CONTRACT_STATUS = 'Active')
  # 	    THEN 1 ELSE 0
  # 	  end isActive
  # 	FROM
  # 	  POS_CASH_balance
  # 	group BY
  # 	  SK_ID_PREV
  #   )
  # select
  #   app.SK_ID_CURR, IFNULL(sum(current_poscash.isActive), 0) AS pos_count_active
  # from
  #   application_train app
  # inner join current_poscash
  #   on app.SK_ID_CURR = current_poscash.SK_ID_CURR
  # GROUP BY
  #   app.SK_ID_CURR
  # ORDER BY
  #   app.SK_ID_CURR

  return train, test, y, db_conn, folds, cache_file
