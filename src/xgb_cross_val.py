# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

import numpy as np
import xgboost as xgb
import time

#Ideally cross val split should be done before feature engineering, and feature engineering + selection should be done separately for each splits so it better mimics out-of-sample predictions
def xgb_cross_val(params, X, y, folds, metric):
    n = 1
    num_rounds = 3000

    list_rounds = []
    list_scores = []

    for train_idx, valid_idx in folds:
        print('#################################')
        print('#########  Validating for fold:', n)

        xgtrain = xgb.DMatrix(X.values[train_idx], label=y.values[train_idx])
        xgtest = xgb.DMatrix(X.values[valid_idx], label=y.values[valid_idx])

        watchlist = [ (xgtest, 'test') ]
        model = xgb.train(params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=False)

        rounds = model.best_ntree_limit
        score = model.best_score

        print('\nFold', n,'- best round:', rounds)
        print('Fold', n,'- best score:', score)

        list_rounds.append(rounds)
        list_scores.append(score)
        n += 1

    mean_score = np.mean(list_scores)
    std_score = np.std(list_scores)
    mean_round = np.mean(list_rounds)
    std_round = np.std(list_rounds)
    print('End cross validating',n-1,'folds') #otherwise it displays 6 folds
    print("Cross Validation Scores are: ", np.round(list_scores,3))
    print("Mean CrossVal score is: ", round(mean_score,3))
    print("Std Dev CrossVal score is: ", round(std_score,3))
    print("Cross Validation early stopping rounds are: ", np.round(list_rounds,3))
    print("Mean early stopping round is: ", round(mean_round,3))
    print("Std Dev early stopping round is: ", round(std_round,3))

    with open('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str(metric)+'-xgb-cv.txt', 'a') as out_cv:
        out_cv.write("Cross Validation Scores are: " + str(np.round(list_scores,3)) + "\n")
        out_cv.write("Mean CrossVal score is: " + str(round(mean_score,3)) + "\n")
        out_cv.write("Std Dev CrossVal score is: " + str(round(std_score,3)) + "\n")
        out_cv.write("Cross Validation early stopping rounds are: " + str(np.round(list_rounds,3)) + "\n")
        out_cv.write("Mean early stopping round is: " + str(round(mean_round,3)) + "\n")
        out_cv.write("Std Dev early stopping round is: " + str(round(std_round,3)) + "\n")

    return mean_round
