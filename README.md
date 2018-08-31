# Home Credit default risk

This is code I built for the [Home Credit default risk competition on Kaggle.](https://www.kaggle.com/c/home-credit-default-risk) This should be seen more as an ML engineering achievement than a data science top of the line prediction model.

# What will you find in the repo?

First of all, due to time constraints this is not a top scorer. First rank was 0.80570 AUC (499 submissions), this is 0.78212 AUC (12 submissions).

However you will hopefully find an inspiring way to architecture your ML project in a fast, scalable and maintainable way. It also demonstrate how to to machine learning on top of existing SQL databases instead of loading everything in Pandas.

# Teasers

![](img/teaser_feat_eng.png)
![](img/teaser_cross_val.png)
![](img/teaser_preditions_feat_importance.png)

# Table of Contents

If you're a Kaggler and just want to run the code go to Starting point. Otherwise, skip the starting point and go directly to the competition description.
- [Home Credit default risk](#home-credit-default-risk)
- [What will you find in the repo?](#what-will-you-find-in-the-repo)
- [Teasers](#teasers)
- [Table of Contents](#table-of-contents)
- [Starting point](#starting-point)
  - [Adding new features](#adding-new-features)
  - [Feature selection](#feature-selection)
- [Description of the competition](#description-of-the-competition)
- [Fast to run and iterate](#fast-to-run-and-iterate)
  - [SQL](#sql)
  - [Caching](#caching)
  - [Comparision with Sklearn pipelines](#comparision-with-sklearn-pipelines)
- [Maintainable](#maintainable)
  - [Simple clean code](#simple-clean-code)
  - [Keep track of experiments](#keep-track-of-experiments)
    - [Features by importance](#features-by-importance)
    - [Features ordered by features](#features-ordered-by-features)
- [Scalable](#scalable)
- [Conclusion](#conclusion)

# Starting point

The code expects all CSVs to be stored in a SQlite database in `./inputs/inputs.db`.
Data is available on [Kaggle Home Credit default risk competition](https://dbeaver.io/):

I used csvs-to-sqlite https://github.com/simonw/csvs-to-sqlite, which is built on top of Pandas. Note that the new [DBeaver 5.1.5](https://dbeaver.io/) added CSV import, it's my recommended SQL browser in any case.

Expected table names are:
  - application_train
  - application_test
  - previous_application
  - bureau
  - bureau_balance
  - credit_card_balance
  - installments_payments
  - POS_CASH_balance

and optionnally:
  - HomeCredit_columns_description
  - sample_submission

Run `python m000_xgboost_baseline.py` to get a baseline score with just `application_train`/`application_test`

Run `python m100_predictions.py` to see the full pipeline in action.

## Adding new features

If you want to edit a feature engineering step:

  - Plain feature extraction is done in the files in `./src/feature_extraction`
  - Feature engineering in `./src/feature_engineering`

Each step accepts:
  - train data
  - test data
  - labels/target
  - database connection
  - your folds
  - a cache file

If you want to add a complete new step, follow one of the already existing examples.

The framework offers the following features:

- `@logspeed` decorator to log your step speed
- `logger = logging.getLogger("HomeCredit")` to load the commandline and CSV logger and display custom messages.
- Cache the train and test results of your step. If you want to clear the cache edit and run `./del_from_cache.py`
- You can use the `y` target labels and `folds` for advanced out-of-folds predictions. I recommend you cache them afterwards.
  An example can be found in a very old version of this pipeline: https://github.com/mratsim/Apartment-Interest-Prediction/blob/master/src/transformers_nlp_tfidf.py#L42

## Feature selection

In this example we built features from scratch but this framework also offers advanced feature selection similar to the `sklearn-pandas` package. See there: https://github.com/mratsim/meilleur-data-scientist-france-2018/blob/master/m110_feat_eng.py#L17-L36
  ```Python
  select_feat = [
    ("encoded_etat", None),
    ("encoded_nom_magasin", None),
    ("prix", None),
    ("nb_images", None),
    ("longueur_image", None),
    ("largeur_image", None),
    ("poids", None),
    ("encoded_categorie", None),
    ("encoded_sous_categorie_1", None),
    ("encoded_sous_categorie_2", None),
    ("encoded_sous_categorie_3", None),
    ("encoded_sous_categorie_4", None),
    ("description_produit", [TfidfVectorizer(max_features=2**16,
                             min_df=2, stop_words='english',
                             use_idf=True),
                             TruncatedSVD(2)]),
    ("encoded_couleur", None),
    ("vintage", None)
  ]
  ```
  and https://github.com/mratsim/meilleur-data-scientist-france-2018/blob/master/m000_baseline.py#L85
  ```Python
  X_train, X_test = feat_selection(select_feat, X, X_test, y)
  ```

  The `None` can be any Sklearn `Rescaler`, `CategoricalEncoder` or even a TfIDF + TruncatedSVD Sklearn pipeline

# Description of the competition

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

![Home Credit Group](./img/about-us-home-credit.jpg)

[Home Credit](http://www.homecredit.net/) strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

# Fast to run and iterate

## SQL

SQL databases are very optimized, multithreaded when critical, with index for O(log N) operations on critical path and operations on multiple fields can be grouped. In entreprise environment, SQL databases are also often installed on beefy servers.

⚠ Caution, spare your support team and DBA and don't run heavy SQL queries on production databases.

Furthermore, you can do feature engineering with any SQL editor with advanced previews, multiple panes:

![](img/sql_workspace.png)

## Caching

Every costly step of the pipeline can be cached using `shelve` from the Python standard library. This avoid having to recompute everything for small changes.

![](img/caching.png)

## Comparision with Sklearn pipelines

While I started machine learning, I was pretty excited with Sklearn pipelines but after tinkering with them a lot they unfortunately came up short on many front:

- No caching
- No support for early stopping
- No out-of-fold predictions
- Unnecessary computation across folds:
  - To avoid leaks on aggregate functions like "mean" or "median" based on target classes they make sure that only the current fold is retrieved and used in computation.
  - In practice those are rare cases, you should bulk retrieve everything and do proper target encoding/crossval predictions on a as needed basis. On a 5 fold CV this will have 5 times less overhead.
- Procedural API/everything is a function (instead of a class), in theory the functions can be run in parallel (see the bottom of star_command.py)

# Maintainable

## Simple clean code

The code to achieve this is extremely simple, for example the credit card balance feature engineering is just the following:

- logger
- `@logspeed` decorator to log the speed of the feature engineering step
- keys in the `shelve` cache database
- the SQL

```Python
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
```

## Keep track of experiments

![](img/output.png)

### Features by importance

![](img/feat_importance.png)

### Features ordered by features

This makes it easy to use diff to check which features were added/removed and their change of influence on the final model across experiments

![](img/feat_importance_diff.png)

# Scalable

1. All the instrumentation provided makes it much easier to keep track on what is going on and not have the dreaded "oh what did I change, I had a much better results previously".
  It also makes it much easier to collaborate on the same model.

2. Pipeline transformations are all done in a [single file](m110_feat_engineering_pipeline.py) and can be easily commented in or out:

    ```Python
    from src.star_command import feat_engineering_pipe
    from src.feature_engineering.fte_money import fte_income_ratios, fte_goods_price
    from src.feature_engineering.fte_cyclic_time import fte_cyclic_time
    from src.feature_engineering.fte_age import fte_age
    from src.feature_engineering.fte_money_bureau import fte_bureau_credit_situation
    from src.feature_engineering.fte_prev_app import fte_prev_credit_situation, fte_prev_app_process, fte_sales_channels
    from src.feature_engineering.fte_credit_balance import fte_withdrawals
    from src.feature_engineering.fte_pos_cash import fte_pos_cash_aggregate, fte_pos_cash_current_status
    from src.feature_engineering.fte_installment_pmt import fte_missed_installments

    from src.feature_extraction.fte_application import fte_application, fte_app_categoricals

    pipe_transforms = feat_engineering_pipe(
      fte_application,
      fte_app_categoricals,
      fte_income_ratios,
      fte_cyclic_time,
      fte_goods_price,
      fte_age,
      fte_prev_credit_situation,
      fte_bureau_credit_situation,
      fte_prev_app_process,
      fte_sales_channels,
      fte_withdrawals,
      fte_pos_cash_aggregate,
      fte_pos_cash_current_status,
      fte_missed_installments
    )
    ```

3. It supports sparse dataframes and being SQL based means that this can be done on big data and clusters.

# Conclusion

I hope you learned something with this repo. Feel free to ask questions in Github issues.
