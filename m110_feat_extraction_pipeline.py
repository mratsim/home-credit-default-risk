# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

from src.star_command import feat_extraction_pipe
from src.feature_engineering.fte_money import fte_income_ratios, fte_goods_price
from src.feature_engineering.fte_cyclic_time import fte_cyclic_time
from src.feature_engineering.fte_age import fte_age
from src.feature_engineering.fte_money_bureau import fte_prev_credit_situation

from src.feature_extraction.fte_employment import fte_organisation
from src.feature_extraction.fte_ext_source import fte_ext_source
from src.feature_extraction.fte_family import fte_family_situation
from src.feature_extraction.fte_credit_inquiries import fte_credit_inquiries
from src.feature_extraction.fte_region import fte_region

pipe_transforms = feat_extraction_pipe(
  fte_income_ratios,
  fte_cyclic_time,
  fte_goods_price,
  fte_age,
  fte_organisation,
  fte_prev_credit_situation,
  fte_ext_source,
  fte_family_situation,
  fte_prev_credit_situation,
  fte_region,
  fte_credit_inquiries
)
