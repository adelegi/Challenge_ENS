import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import re
from tqdm import tqdm 
import sklearn

import data
import features


class ModelLearning():
    def __init__(self, features, output, type_features=''):
        self.features = features
        self.output = output
        self.output_names = list(self.output['fields'].keys())

        self.name_buildings = list(self.features.keys())
        self.name_metrics = ['mse', 'mse pct']

        self.sort_settings_by_building()

    def sort_settings_by_building(self):
        """  """
        self.sort_settings_building = {}

        for building in self.name_buildings:
            num_building = [int(x[1:]) for x in  re.findall("_[0-9]*", building)][0]

            if num_building not in self.sort_settings_building:
                self.sort_settings_building[num_building] = []

            self.sort_settings_building[num_building].append(building)

    def load_train_test_set(self, features_names=None, pct_train=0.8, do_print=False):
        """ Separate the data in a test set and a train set 
            output_var = list of the variables to load
        """

        if features_names is None:
            features_names = list(self.features[self.name_buildings[0]].columns)
    
        nb_buildings = len(self.sort_settings_building)
        nb_buildings_train = int(pct_train * nb_buildings)  # nb of buildings in train set

        if do_print:
            print("{} buildings with {} settings each in the test set, {} in the train set"\
                  .format(nb_buildings - nb_buildings_train, len(self.sort_settings_building[10]), nb_buildings_train))
        
        indices = np.random.permutation(range(nb_buildings))

        self.X_train =  np.array([], dtype=np.int64).reshape(0, len(features_names))
        self.Y_train = np.array([], dtype=np.int64).reshape(0, 5)
        self.X_test = np.array([], dtype=np.int64).reshape(0, len(features_names))
        self.Y_test = np.array([], dtype=np.int64).reshape(0, 5)

        test_building = []
        
        # Train sets
        for i in indices[1:nb_buildings_train]:
            for setting in self.sort_settings_building[i+1]:
                self.X_train = np.concatenate((self.X_train, self.features[setting][features_names]), axis=0)
                self.Y_train = np.concatenate((self.Y_train, self.output[setting]), axis=0)

        # Test sets
        for i in indices[nb_buildings_train:-1]:
            for setting in self.sort_settings_building[i+1]:
                test_building.append(setting)
                self.X_test = np.concatenate((self.X_test, self.features[setting][features_names]), axis=0)
                self.Y_test = np.concatenate((self.Y_test, self.output[setting]), axis=0)

    def fit_model(self, var):
        """ Fit the model to the train set """
        self.var = var
        self.col_var = self.output['fields'][var]

        pass

    def predict_model(self, model, X):
        """ Return Y predicted by the model from X data """
        pass

    def test_model(self):
        """ Test the model on the test set and return performance metrics """

        Y_pred = self.predict_model(self.model, self.X_test)
        mse = sklearn.metrics.mean_squared_error(self.Y_test[:, self.col_var], Y_pred)

        Y_divided = [x if x !=0 else 1 for x in self.Y_test[:, self.col_var]]
        mse_pct = np.mean(((Y_pred - self.Y_test[:, self.col_var]) / Y_divided)**2)

        return {'mse': mse, 'mse pct': mse_pct}

    def cross_validate(self, features_names, var_names=None, N=1, pct_train=0.75, do_print=False):
        """ For each variable in var_names, compute N times : separate test/train data,
            train on train data to measure metrics on test data """

        if var_names is None:
            var_names = list(self.output['fields'].keys())

        # Init variables
        results = {}
        for var in var_names:
            results[var] = {}
            for metric in self.name_metrics:
                results[var][metric] = []

        # Cross-validation
        print("Evolution of the {} iterations:".format(N))
        for n in tqdm(range(N)):
            self.load_train_test_set(features_names, pct_train)

            for var in var_names:
                self.fit_model(var)

                result_metrics = self.test_model()
                for metric in self.name_metrics:
                    results[var][metric].append(result_metrics[metric])

        # Print
        if do_print:
            for var in var_names:
                for metric in self.name_metrics:
                    avg_metric = np.mean(results[var][metric])
                    print("Average {} of {}: {}".format(metric, var, avg_metric))

        return results

    def save_output(self, model_dico, file_save, X_val):
        """ Save output predictions in the right format
              - model_dico: dictionary of the trained models for each output """

        pred = {}

        # Prediction
        for o in self.output_names:
            pred[o] = self.predict_model(model_dico[o] , X_val)

        # Text to save
        output_text = ':'.join(self.output_names)
        
        for i in tqdm(range(len(pred[self.output_names[0]]))):
            output_text += '\n'
            output_text += ':'.join([str(pred[o][i]) for o in self.output_names])
            
        with open(file_save, 'w') as file:
            file.write(output_text)

        print("File saved at '{}'".format(file_save))

    def train_model_to_plot(self, name_output):
        self.name_output = name_output
        self.col_var = self.output['fields'][name_output]

        # Fit model
        self.fit_model(name_output)

        self.Y_pred = self.predict_model(self.model, self.X_test)

    def plot_period(self, n1, n2):
        plt.plot(self.Y_pred[n1:n2], label='prediction')
        plt.plot(self.Y_test[n1:n2, self.col_var], label='ground truth')
        plt.title("Prediction of '{}'".format(self.name_output))
        plt.xlabel('Hour')
        plt.ylabel(self.name_output)
        plt.legend()
        plt.show()

        diff = self.Y_pred[n1:n2] - self.Y_test[n1:n2, self.col_var]
        plt.plot(diff)
        plt.title("Difference between the ground truth and the prediction of '{}'"\
                  .format(self.name_output))
        plt.xlabel('Hour')
        plt.show()