# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd

def fte_prev_credit_situation(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      IFNULL(sum(case(p.NAME_CONTRACT_TYPE) when 'Consumer loans' then 1 else 0 end), 0) AS prev_total_consumer_loans,
      IFNULL(sum(case(p.NAME_CONTRACT_TYPE) when 'Cash loans' then 1 else 0 end), 0) AS prev_total_cash_loans,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Refused' then 1 else 0 end), 0) AS prev_total_application_refused,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Canceled' then 1 else 0 end), 0) AS prev_total_application_canceled,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Approved' then 1 else 0 end), 0) AS prev_total_application_approved,
      IFNULL(sum(case(p.NAME_CONTRACT_STATUS) when 'Unused offer' then 1 else 0 end), 0) AS prev_total_application_unused,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Refused' then p.AMT_APPLICATION else 0 end), 0) AS prev_avg_application_refused,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Canceled' then p.AMT_APPLICATION else 0 end), 0) AS prev_avg_application_canceled,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Approved' then p.AMT_APPLICATION else 0 end), 0) AS prev_avg_application_approved,
      IFNULL(avg(case(p.NAME_CONTRACT_STATUS) when 'Unused offer' then p.AMT_APPLICATION else 0 end), 0) AS prev_avg_application_unused,
      avg(case(p.NAME_CONTRACT_STATUS) when 'Approved' then 1.0 when 'Unused offer' then 1.0 else 0.0 end) AS prev_ratio_app_approved_total,
      IFNULL(avg(p.AMT_APPLICATION - p.AMT_CREDIT), 0) AS prev_avg_diff_asked_offered,
      IFNULL(avg(p.CNT_PAYMENT), 0) AS prev_avg_payment_schedule,
      IFNULL(avg(p.AMT_ANNUITY), 0) AS prev_avg_annuity,
      IFNULL(avg(p.CNT_PAYMENT), 0) / app.AMT_ANNUITY AS prev_ratio_annuity_prev_app,
      avg(p.AMT_DOWN_PAYMENT) AS prev_avg_down_payment
    from
      {table} app
    left join
      previous_application p on app.SK_ID_CURR = p.SK_ID_CURR
    GROUP BY
      app.SK_ID_CURR
    ORDER BY
      app.SK_ID_CURR ASC;
    """

    df[['prev_total_consumer_loans',
        'prev_total_cash_loans',
        'prev_total_application_refused',
        'prev_total_application_canceled',
        'prev_total_application_approved',
        'prev_total_application_unused',
        'prev_avg_application_refused',
        'prev_avg_application_canceled',
        'prev_avg_application_approved',
        'prev_avg_application_unused',
        'prev_ratio_app_approved_total',
        'prev_avg_diff_asked_offered',
        'prev_avg_payment_schedule',
        'prev_avg_annuity',
        'prev_ratio_annuity_prev_app',
        'prev_avg_down_payment']] = pd.read_sql_query(query, db_conn)

    # TODO add currency, otherwise credit is not comparable

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file

def fte_prev_app_process(train, test, y, db_conn, folds, cache_file):
  def _trans(df, table):
    query = f"""
    select
      avg(-p.DAYS_DECISION) / 365.25 prev_avg_years_decision,
      avg(case(p.NAME_CONTRACT_TYPE)
        when 'Revolving loans' Then NULL
        else -p.DAYS_FIRST_DUE
        end
      ) / 365.25 prev_avg_years_first_due,
      avg(case(p.NAME_CONTRACT_TYPE)
        when 'Revolving loans' Then NULL
        else -p.DAYS_LAST_DUE_1ST_VERSION
        end
      ) / 365.25 prev_avg_years_last_due_1st_version,
      avg(p.HOUR_APPR_PROCESS_START) prev_avg_hour_start,
      sum(case(p.NAME_CONTRACT_TYPE)
        when 'Revolving loans' Then 1
        else 0
        end
      ) prev_count_revolving_loans
    from
      application_train app
    left join
      previous_application p on app.SK_ID_CURR = p.SK_ID_CURR
    GROUP BY
      app.SK_ID_CURR
    ORDER BY
      app.SK_ID_CURR ASC;
    """

    df[['prev_avg_years_decision',
        'prev_avg_years_first_due',
        'prev_avg_years_last_due_1st_version',
        'prev_avg_hour_start',
        'prev_count_revolving_loans'
        ]] = pd.read_sql_query(query, db_conn)

    # TODO add currency, otherwise credit is not comparable

  _trans(train, "application_train")
  _trans(test, "application_test")

  return train, test, y, db_conn, folds, cache_file
