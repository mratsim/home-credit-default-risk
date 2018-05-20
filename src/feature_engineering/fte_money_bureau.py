# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_bureau_credit_situation(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      IFNULL(count(b.SK_ID_CURR), 0) AS bureau_total_prev_applications,
      IFNULL(sum(case CREDIT_ACTIVE when 'Active' then 1 else 0 end), 0) AS bureau_current_active_applications,
      IFNULL(sum(AMT_CREDIT_SUM), 0) AS bureau_total_prev_credit,
      IFNULL(sum(case CREDIT_ACTIVE when 'Active' then AMT_CREDIT_SUM else 0 end), 0) AS bureau_active_credit_amount,
      IFNULL(sum(AMT_CREDIT_SUM_DEBT), 0) AS bureau_current_debt,
      IFNULL(sum(AMT_CREDIT_MAX_OVERDUE), 0) AS bureau_current_overdue,
      IFNULL(max(-DAYS_CREDIT), 99 * 365.25) / 365.25 AS bureau_first_credit_years_ago,
      IFNULL(min(-DAYS_CREDIT), 99 * 365.25) / 365.25 AS bureau_last_credit_years_ago,
      IFNULL(max(DAYS_CREDIT_ENDDATE), -99 * 365.25) / 365.25 AS bureau_existing_credit_close_date,
      IFNULL(max(-DAYS_ENDDATE_FACT), 99 * 365.25) / 365.25 AS bureau_years_since_no_card_credit
    from
      {table} app
    left join
      bureau b on app.SK_ID_CURR = b.SK_ID_CURR
    GROUP BY
      app.SK_ID_CURR
    ORDER BY
      app.SK_ID_CURR ASC;
    """

    df[['bureau_total_prev_applications',
        'bureau_current_active_applications',
        'bureau_total_prev_credit',
        'bureau_active_credit_amount',
        'bureau_current_debt',
        'bureau_current_overdue',
        'bureau_first_credit_years_ago',
        'bureau_last_credit_years_ago',
        'bureau_existing_credit_close_date',
        'bureau_years_since_no_card_credit'
        ]] = pd.read_sql_query(query, db_conn)

    # TODO add currency, otherwise credit is not comparable

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
