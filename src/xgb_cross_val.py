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

    mean_round = np.mean(list_rounds)

    str_mean_score = '{:.4f}'.format(np.mean(list_scores))
    str_std_score = '{:.4f}'.format(np.std(list_scores))
    str_mean_round = '{:.4f}'.format(mean_round)
    str_std_round = '{:.4f}'.format(np.std(list_rounds))
    print('End cross validating',n-1,'folds') #otherwise it displays 6 folds
    print("Cross Validation Scores are: ", np.round(list_scores,4))
    print("Mean CrossVal score is: ", str_mean_score)
    print("Std Dev CrossVal score is: ", str_std_score)
    print("Cross Validation early stopping rounds are: ", np.round(list_rounds,4))
    print("Mean early stopping round is: ", str_mean_round)
    print("Std Dev early stopping round is: ", str_std_round)

    str_metric = '{:.4f}'.format(round(metric,4))

    with open('./outputs/'+time.strftime("%Y-%m-%d_%H%M-")+'-valid'+str_metric+'-xgb-cv.txt', 'a') as out_cv:
        out_cv.write("Cross Validation Scores are: " + str(np.round(list_scores,4)) + "\n")
        out_cv.write("Mean CrossVal score is: " + str_mean_score + "\n")
        out_cv.write("Std Dev CrossVal score is: " + str_std_score + "\n")
        out_cv.write("Cross Validation early stopping rounds are: " + str(np.round(list_rounds,4)) + "\n")
        out_cv.write("Mean early stopping round is: " + str_mean_round + "\n")
        out_cv.write("Std Dev early stopping round is: " + str_std_round + "\n")

    return mean_round
