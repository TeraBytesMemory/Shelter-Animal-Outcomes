#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import csv
import numpy as np

from model import train_data_format, test_data_format, data_format
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, tpe, hp, rand
from sklearn import cross_validation


if __name__ == '__main__':
    parameters = {
        'n_estimators': [100, 250, 500, 1000],
        'max_depth': [3, 5, 10, 15, 20, 25, 30],
        'reg_alpha': [3.0, 4.0],
        'missing': [np.nan, 1.0, 2.0, 3.0]
    }

    hp_choice = dict([(key, hp.choice(key, value))
                      for key, value in parameters.items()])


    # train
    ''''
    df = data_format('train.csv')
    df['OutcomeType'] = df['OutcomeType'].map({
        'Adoption': 0 , 'Died': 1, 'Euthanasia': 2,
        'Return_to_owner': 3, 'Transfer': 4
    })
    y = df['OutcomeType'].values

    df = df.drop(['AnimalID', 'OutcomeType', 'OutcomeSubtype'], axis=1)
    X = df.values

    #df_log = df.drop(['DateTime', 'Name', 'SexuponOutcome', 'AgeuponOutcome'], axis=1)
    df_log = df
    #rf_args = {'random_state': 0, 'max_features': 3, 'min_samples_split': 5,
    #           'max_depth': 10, 'n_jobs': 2, 'n_estimators': 250}
    '''

    X, y = train_data_format('train.csv')
    lr = RandomForestClassifier()
    lr.fit(X, y)
    new_feature = lr.predict(X)
    X = np.hstack((X, new_feature[np.newaxis].T))

    def estimator(args):
        print("Args:", args)
        forest = xgb.XGBClassifier(**args)

        trainX, testX, trainy, testy = cross_validation.train_test_split(
            X, y, test_size=0.4, random_state=0)

        forest.fit(trainX, trainy)

        #    print 'Predicting...'

        acu = log_loss(forest, testX, testy)
        #acu = forest.score(testX, testy)

        print("Accurate:", acu)
        return -acu


    best = fmin(estimator, hp_choice, algo=tpe.suggest, max_evals=5)
    best = dict([(key, parameters[key][value]) for key, value in best.items()])

    #print("\nBest Model...")
    #estimator(best)

    clf = xgb.XGBClassifier(**best)
    clf.fit(X, y)

    print(clf.booster().get_fscore())

    '''
    #test_data, ids = test_data_format('test.csv')
    test_df = data_format('test.csv', train=False)

    ids = test_df['ID'].values
    df = test_df.drop(['ID'], axis=1)
    test_data = test_df.values

    #df_log = test_df.drop(['DateTime', 'Name', 'SexuponOutcome', 'AgeuponOutcome'], axis=1)
    df_log = test_df
    '''

    test_data, ids = test_data_format('test.csv')
    new_feature = lr.predict(test_data)
    test_data = np.hstack((test_data, new_feature[np.newaxis].T))

    result = clf.predict_proba(test_data)
    result = result.astype(np.object)

    ids = ids.astype(int)
    output = np.hstack((ids[np.newaxis].T, result))

    predictions_file = open("submission.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['ID', 'Adoption', 'Died',
                               'Euthanasia', 'Return_to_owner', 'Transfer'])
    open_file_object.writerows(output)
    predictions_file.close()
    print('Done.')
