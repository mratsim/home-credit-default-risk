# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

from src.star_command import feat_extraction_pipe
from src.feature_engineering.fte_income_ratios import fte_income_ratios
from src.feature_engineering.fte_cyclic_time import fte_cyclic_time

pipe_transforms = feat_extraction_pipe(
  fte_income_ratios,
  fte_cyclic_time
)
