# Delete a key from the cache file

from src.cache import del_from_cache

del_from_cache('./cache.db',
  [
    'fte_missed_installments_train',
    'fte_missed_installments_test'
  ])
