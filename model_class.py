import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import re
from tqdm import tqdm 

import data
import features


class ModelLearning():
    def __init__(self, features, output):
        self.features = features
        self.output = output
        self.output_names = list(self.output['fields'].keys())

        self.name_buildings = list(self.features.keys())

        self.sort_settings_by_building()

    def sort_settings_by_building(self):
        """  """
        self.sort_settings_building = {}

        for building in self.name_buildings:
            num_building = [int(x[1:]) for x in  re.findall("_[0-9]*", building)][0]

            if num_building not in self.sort_settings_building:
                self.sort_settings_building[num_building] = []

            self.sort_settings_building[num_building].append(building)

    def load_train_test_set(self, output_var, features_names=None, pct_train=0.8):
        """ Separate the data in a test set and a train set 

            TODO : Separer par building et non pas par setting"""

        if features_names is None:
            features_names = list(self.features[self.name_buildings[0]].columns)
    
        num_output = self.output['fields'][output_var]
        nb_buildings = len(self.name_buildings)
        nb_buildings_train = int(pct_train * nb_buildings)  # nb of buildings in train set
        
        indices = np.random.permutation(range(nb_buildings))
        
        building_0 = self.name_buildings[indices[0]]
        self.X_train = self.features[building_0][features_names]
        self.Y_train = self.output[building_0][:, self.output['fields'][output_var]]

        last_building = self.name_buildings[indices[-1]]
        self.X_test = self.features[last_building][features_names]
        self.Y_test = self.output[last_building][:, self.output['fields'][output_var]]
        test_building = []
        
        # Train sets
        for i in indices[1:nb_buildings_train]:
            building = self.name_buildings[i]
            self.X_train = np.concatenate((self.X_train, self.features[building][features_names]), axis=0)
            self.Y_train = np.concatenate((self.Y_train, self.output[building][:, num_output]), axis=0)

        # Test sets
        for i in indices[nb_buildings_train:-1]:
            building = self.name_buildings[i]
            test_building.append(building)
            self.X_test = np.concatenate((self.X_test, self.features[building][features_names]), axis=0)
            self.Y_test = np.concatenate((self.Y_test, self.output[building][:, num_output]), axis=0)

    def fit_model(self):
        """ Fit the model to the train set """
        pass

    def test_model(self):
        """ Test the model on the test set and return performance metrics """
        pass

    def predict_model(self, model, X):
        """ Return Y predicted by the model from X data """
        pass

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

        # Fit model
        self.load_train_test_set(name_output, pct_train=0.8)
        self.fit_model()

        self.Y_pred = self.predict_model(self.model, self.X_test)

    def plot_period(self, n1, n2):
        plt.plot(self.Y_pred[n1:n2], label='prediction')
        plt.plot(self.Y_test[n1:n2], label='ground truth')
        plt.title("Prediction of '{}'".format(self.name_output))
        plt.xlabel('Hour')
        plt.ylabel(self.name_output)
        plt.legend()
        plt.show()

        diff = self.Y_pred[n1:n2] - self.Y_test [n1:n2]
        plt.plot(diff)
        plt.title("Difference between the ground truth and the prediction of '{}'"\
                  .format(self.name_output))
        plt.xlabel('Hour')
        plt.show()