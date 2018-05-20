# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_prev_credit_situation(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      IFNULL(sum(case(p.NAME_CONTRACT_TYPE) when 'Consumer loans' then 1 else 0 end), 0) AS total_consumer_loans,
      IFNULL(sum(case(p.NAME_CONTRACT_TYPE) when 'Cash loans' then 1 else 0 end), 0) AS total_cash_loans,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Refused' then 1 else 0 end), 0) AS total_application_refused,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Canceled' then 1 else 0 end), 0) AS total_application_canceled,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Approved' then 1 else 0 end), 0) AS total_application_approved,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Unused offer' then 1 else 0 end), 0) AS total_application_unused,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Refused' then p.AMT_APPLICATION else 0 end), 0) AS avg_application_refused,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Canceled' then p.AMT_APPLICATION else 0 end), 0) AS avg_application_canceled,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Approved' then p.AMT_APPLICATION else 0 end), 0) AS avg_application_approved,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Unused offer' then p.AMT_APPLICATION else 0 end), 0) AS avg_application_unused,
      avg(case(p.NAME_CONTRACT_STATUS) when 'Approved' then 1.0 when 'Unused offer' then 1.0 else 0.0 end) AS ratio_app_approved_total,
      IFNULL(avg(p.AMT_APPLICATION - p.AMT_CREDIT), 0) AS avg_diff_asked_offered
      IFNULL(avg(p.CNT_PAYMENT), 0) AS avg_payment_schedule
      IFNULL(avg(p.ANNUITY), 0) AS avg_annuity
      IFNULL(avg(p.CNT_PAYMENT), 0) / app.ANNUITY AS ratio_annuity_prev_app
    from
      {table} app
    left join
      previous_application p on app.SK_ID_CURR = p.SK_ID_CURR
    GROUP BY
      app.SK_ID_CURR
    ORDER BY
      app.SK_ID_CURR ASC;
    """

    df[['total_consumer_loans',
        'total_cash_loans',
        'total_application_refused',
        'total_application_canceled',
        'total_application_approved',
        'total_application_unused',
        'avg_application_refused',
        'avg_application_canceled',
        'avg_application_approved',
        'avg_application_unused',
        'ratio_app_approved_total',
        'avg_diff_asked_offered',
        'avg_payment_schedule',
        'avg_annuity',
        'ratio_annuity_prev_app']] = pd.read_sql_query(query, db_conn)

    # TODO add currency, otherwise credit is not comparable

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
