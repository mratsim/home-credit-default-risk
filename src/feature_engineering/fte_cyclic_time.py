# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd
import numpy as np
from src.instrumentation import logspeed

@logspeed
def fte_cyclic_time(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      --case(WEEKDAY_APPR_PROCESS_START)
      --  when "SATURDAY" THEN 1
      --  when "SUNDAY" THEN 1
      --  else 0
      --end isOccidentalWeekend,
      case(WEEKDAY_APPR_PROCESS_START)
        when "MONDAY" THEN 0
        when "TUESDAY" THEN 1
        when "WEDNESDAY" THEN 2
        when "THURSDAY" THEN 3
        when "FRIDAY" THEN 4
        when "SATURDAY" THEN 5
        when "SUNDAY" THEN 6
        else "NA"
      end dayOfWeek,
      HOUR_APPR_PROCESS_START
    from
      {table}
    order by
      SK_ID_CURR ASC;
    """

    df[[
      # 'isOccidentalWeekend',
      'dayOfWeek',
      'HOUR_APPR_PROCESS_START']] = pd.read_sql_query(query, db_conn)
    # df['cos_dayOfWeek'] = np.cos(df['dayOfWeek'] / 7 * 2 * np.pi)
    # df['sin_dayOfWeek'] = np.sin(df['dayOfWeek'] / 7 * 2 * np.pi)
    # df['cos_hour_start'] = np.cos(df['HOUR_APPR_PROCESS_START'] / 24 * 2 * np.pi)
    # df['sin_hour_start'] = np.sin(df['HOUR_APPR_PROCESS_START'] / 24 * 2 * np.pi)

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
