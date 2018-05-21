# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import numpy as np
import xgboost as xgb
import time
import logging

## Get the same logger from main"
logger = logging.getLogger("HomeCredit")

#Ideally cross val split should be done before feature engineering, and feature engineering + selection should be done separately for each splits so it better mimics out-of-sample predictions
def xgb_cross_val(params, X, y, folds, metric):
    n = 1
    num_rounds = 3000

    list_rounds = []
    list_scores = []

    for train_idx, valid_idx in folds:
        logger.info('#################################')
        logger.info(f'#########  Validating for fold: {n}')

        xgtrain = xgb.DMatrix(X.values[train_idx], label=y.values[train_idx])
        xgtest = xgb.DMatrix(X.values[valid_idx], label=y.values[valid_idx])

        watchlist = [ (xgtest, 'test') ]
        model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=False)

        rounds = model.best_ntree_limit
        score = model.best_score

        logger.info(f'Fold {n} - best round: {rounds}')
        logger.info(f'Fold {n} - best score: {score}\n')

        list_rounds.append(rounds)
        list_scores.append(score)
        n += 1

    mean_round = np.mean(list_rounds)

    logger.info(f'End cross validating {n-1} folds') #otherwise it displays 6 folds
    logger.info("Cross Validation Scores are: " + str(np.round(list_scores,4)))
    logger.info(f'Mean CrossVal score is:  {np.mean(list_scores):.4f}')
    logger.info(f'Std Dev CrossVal score is: {np.std(list_scores):.4f}')
    logger.info("Cross Validation early stopping rounds are: " + str(np.round(list_rounds,4)))
    logger.info(f'Mean early stopping round is: {mean_round:.4f}')
    logger.info(f'Std Dev early stopping round is: {np.std(list_rounds):.4f}\n')

    return mean_round
