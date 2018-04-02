import numpy as np
import math
import time
import pandas as pd
from tqdm import tqdm
import sklearn
import pickle as pkl
import matplotlib.pyplot as plt

import xgboost as xgb

import data
import features
from model_xgboost import ModelXGBoost


def scores(model):
    # Test Score
    Y_pred = model.predict_model(model.model, model.X_test)
    mse = sklearn.metrics.mean_squared_error(model.Y_test[:, model.col_var], Y_pred)
    print("MSE on test set:", mse)

    # Train score
    Y_pred_train = model.predict_model(model.model, model.X_train)
    mse_train = sklearn.metrics.mean_squared_error(model.Y_train[:, model.col_var], Y_pred_train)
    print("MSE on train set:", mse_train)
    
    return Y_pred

def plot(model, n1, n2, plot_Y):
    plt.plot(plot_Y[n1:n2], label='prediction')
    plt.plot(model.Y_test[n1:n2, model.col_var], label='ground truth')
    plt.legend()
    plt.show()

def test_xgboost(name_features, all_features, output, var,
                 n_estimators=100, max_depth=3, min_child_weight=1, subsample=0.5):
    
    print("-- Variable: '{}', n_estimators={}, max_depth={}".format(var, n_estimators, max_depth))
    
    model = ModelXGBoost(all_features, output)
    model.load_train_test_set(features_names=name_features, pct_train=.8, do_print=False)

    start_time = time.time()
    model.fit_model(var, n_estimators, max_depth, min_child_weight, subsample)
    print("Execution time: {} min".format(round((time.time() - start_time)/60, 2)))

    Y_pred = scores(model)
    
    return model

def predict_val(name_features, all_features, output, var, X_val,
                n_estimators=100, max_depth=3, min_child_weight=1, subsample=0.5):
    model = ModelXGBoost(all_features, output)
    model.load_train_test_set(features_names=name_features, pct_train=1., do_print=False)
    
    start_time = time.time()
    model.fit_model(var, n_estimators, max_depth, min_child_weight, subsample)
    print("Execution time: {} min".format(round((time.time() - start_time)/60, 2)))
    
    Y_pred = model.predict_model(model.model, X_val)
    
    return Y_pred

def test_xgbppst(model, name_features, all_features, output,
                 N, n_estimators_list, max_depth_list, do_print=False):

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for var in output['fields']:
                for i in range(N):
                    model = test_xgboost(name_features, all_features, output, var,
                                         n_estimators, max_depth)
                
                # Affichages
                if do_print:
                    Y_pred_test = model.predict_model(model.model, model.X_test)
                    model.model_importance()
                    num_sem = 5
                    n1, n2 = num_sem*24*7, num_sem*24*7 + 7*24
                    plot(model, n1, n2, Y_pred_test)
                
                Y_pred_val = predict_val(name_features, all_features, output, var, X_val,
                                         n_estimators, max_depth)

                name_svg = './data/Y_pred/Y_val_{}.pkl'.format(var + '_' + t + '_' + str(n_estimators) + str(max_depth))
                pkl.dump(Y_pred_val, open(name_svg, 'wb'))
                print("Prediction saved at " + name_svg)


if __name__ == '__main__':

    # Load features
    temp, dico = data.load_input_data('data/train_input.csv')
    output = data.load_output_data('data/challenge_output.csv', temp, dico)
    all_features = features.load_all_features(dico, temp, remove_useless=True)

    # Choose features
    t =  '_non_int'

    name_features = features.choose_name_features(all_features, t)
    for name in ['surface_2_m2_OUTW', 'surface_3_m2_OUTW', 'surface_4_m2_OUTW',
                 'window_percent_2_outwall', 'window_percent_3_outwall', 'window_percent_4_outwall',
                 'heating_value_t_1', 'AC_value_t_1', 'outside_temp_t_1',
                 #'week', 'week_day'
                ]:
        name_features.remove(name)

    # Load X_val
    X_val = features.load_data_features('./data/test_input.csv', name_features, remove_useless=True)

    n_estimators_list = [1000]
    max_depth_list = [5, 10]
    test_xgbppst(model, name_features, all_features, output,
                 N, n_estimators_list, max_depth_list, do_print=False)