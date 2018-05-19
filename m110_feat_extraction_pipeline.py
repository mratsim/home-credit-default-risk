# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

from src.star_command import feat_extraction_pipe
from src.feature_engineering.fte_income_credit_ratio import fte_credit_income_ratio


pipe_transforms = feat_extraction_pipe(
  fte_credit_income_ratio
)
