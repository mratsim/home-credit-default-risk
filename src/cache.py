# Copyright 2017-2018 Mamy André-Ratsimbazafy. All rights reserved.

import os
import shelve
from pickle import HIGHEST_PROTOCOL

# Logic to load data from cache file

def load_from_cache(cache_file, key_train, key_test):
    if os.path.isfile(cache_file):
        with shelve.open(cache_file, flag='r', protocol=HIGHEST_PROTOCOL) as db:
            if (key_train in db) and (key_test in db):
                dict_train = db[key_train]
                dict_test = db[key_test]
                # db.close()
                return dict_train, dict_test
    return None, None

def save_to_cache(cache_file, key_train, key_test, dict_train, dict_test):
    with shelve.open(cache_file, flag='c', protocol=HIGHEST_PROTOCOL) as db:
        db[key_train] = dict_train
        db[key_test] = dict_test
        # db.close()
