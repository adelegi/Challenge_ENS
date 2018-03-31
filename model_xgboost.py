import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt

from model_class import ModelLearning


class ModelXGBoost(ModelLearning):
    def __init__(self, features, output):
        ModelLearning.__init__(self, features, output)

    def fit_model(self, var):
        """ Fit the model to the train set """

        ModelLearning.fit_model(self, var)

        # Transform data to DGMatrix needed for xgboost

        self.model = xgb.XGBRegressor(silent=False)
        self.model.fit(self.X_train , self.Y_train[:, self.col_var])

    def predict_model(self, model, X):
        """ Return Y predicted by the model from X data """
        return self.model.predict(X)

    def model_importance(self):
        importance = self.model.feature_importances_
        x = range(len(importance))
        plt.bar(x, importance)
        plt.xticks(x, self.features_names, rotation=90)
        plt.title("Feature importance for the variable: '{}'".format(self.var))
        plt.show()