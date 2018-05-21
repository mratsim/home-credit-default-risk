# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

from src.star_command import feat_engineering_pipe
from src.feature_engineering.fte_money import fte_income_ratios, fte_goods_price
from src.feature_engineering.fte_cyclic_time import fte_cyclic_time
from src.feature_engineering.fte_age import fte_age
from src.feature_engineering.fte_money_bureau import fte_bureau_credit_situation
from src.feature_engineering.fte_prev_app import fte_prev_credit_situation, fte_prev_app_process, fte_sales_channels
from src.feature_engineering.fte_credit_balance import fte_withdrawals
from src.feature_engineering.fte_pos_cash import fte_pos_cash_aggregate

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
  fte_pos_cash_aggregate
)
