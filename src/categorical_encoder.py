# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import pandas as pd


def encode_categoricals(df, db_conn, table, field):
  # I don't use Scikit-learn CategoricalEncoder or LabelBinarizer
  # Too much overhead, stateful, slow.
  query_mapper = f"""
  select
    distinct {field}
  from
    {table}
  order by
    ORGANIZATION_TYPE ASC;
  """

  df_mapper = pd.read_sql_query(query_mapper, db_conn)

  # Replace labels by their ID.
  # Note that "Industry: type 10" will before "Industry: type 2"
  dict_mapper = {label: index for (index, label) in df_mapper[field].iteritems()}

  query = f"""
  select
    {field}
  from
    {table}
  order by
    SK_ID_CURR ASC;
  """

  return pd.read_sql_query(query, db_conn)[field].map(dict_mapper)
