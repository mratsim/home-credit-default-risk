# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

from src.star_command import feat_extraction_pipe
from src.feature_engineering.fte_money import fte_income_ratios, fte_goods_price
from src.feature_engineering.fte_cyclic_time import fte_cyclic_time
from src.feature_engineering.fte_age import fte_age

pipe_transforms = feat_extraction_pipe(
  fte_income_ratios,
  fte_cyclic_time,
  fte_goods_price,
  fte_age
)
