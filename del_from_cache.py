# Delete a key from the cache file

from src.cache import del_from_cache

del_from_cache('./cache.db',
  [
    'fte_bureau_credit_situation_train',
    'fte_bureau_credit_situation_test'
  ])
