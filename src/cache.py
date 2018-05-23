# Copyright 2017-2018 Mamy André-Ratsimbazafy. All rights reserved.

import os
import shelve
from pickle import HIGHEST_PROTOCOL

# Logic to load data from cache file

def load_from_cache(cache_file, key_train, key_test):
  if os.path.isfile(cache_file):
    with shelve.open(cache_file, flag='r', protocol=HIGHEST_PROTOCOL) as db:
      if (key_train in db) and (key_test in db):
        train_cached = db[key_train]
        test_cached = db[key_test]
        # db.close()
        return train_cached, test_cached
  return None, None

def save_to_cache(cache_file, key_train, key_test, train_cached, test_cached):
  with shelve.open(cache_file, flag='c', protocol=HIGHEST_PROTOCOL) as db:
    db[key_train] = train_cached
    db[key_test] = test_cached
    # db.close()

def del_from_cache(cache_file, keys_list):
  if os.path.isfile(cache_file):
    with shelve.open(cache_file, flag='w', protocol=HIGHEST_PROTOCOL) as db:
      for key in keys_list:
        del db[key]
    print('FINISHED')
  else:
    print('No file found')
