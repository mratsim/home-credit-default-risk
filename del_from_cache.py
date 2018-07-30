# Delete a key from the cache file

from src.cache import del_from_cache

del_from_cache('./cache.db',
  [
    'fte_withdrawals_train',
    'fte_withdrawals_test'
  ])
