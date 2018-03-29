""" Decision Tree for Regression by skleanr package """

from sklearn.tree import DecisionTreeRegressor

from model_class import ModelLearning

class ModelTreeRegressor(ModelLearning):
    def __init__(self, features, output):
        ModelLearning.__init__(self, features, output)

    def fit_model(self, var):
        """ Fit the model to the train set """

        ModelLearning.fit_model(self, var)

        self.model = DecisionTreeRegressor()
        self.model.fit(self.X_train , self.Y_train[:, self.col_var])

    def predict_model(self, model, X):
        """ Return Y predicted by the model from X data """
        return model.predict(X)