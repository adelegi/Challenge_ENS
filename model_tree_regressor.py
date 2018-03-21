import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time

import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

from model_class import ModelLearning

class ModelTreeRegressor(ModelLearning):
    def __init__(self, features, output):
        ModelLearning.__init__(self, features, output)

    def fit_model(self):
        """ Fit the model to the train set """

        self.model = DecisionTreeRegressor()
        self.model.fit(self.X_train , self.Y_train)

    def test_model(self):
        """ Test the model on the test set and return performance metrics """
        
        Y_pred = self.model.predict(self.X_test)
        return {'mse': sklearn.metrics.mean_squared_error(self.Y_test, Y_pred)}

    def predict_model(self, model, X):
        """ Return Y predicted by the model from X data """
        return model.predict(X)