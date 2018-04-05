import numpy as np
from tqdm import tqdm



def random_drawing(features_dict, output_dict):
    """
    """
    
    train_buildings = np.random.choice(12, 9, replace=False) + 1
    
    X_train =  []
    Y_train = []
    X_test = []
    Y_test = []

    for i in tqdm(range(1, 13)):
        for j in range(20):
            
            name = 'building_{}_{}'.format(i, j)
            features = features_dict[name]
            output = output_dict[name]
        
            if i in train_buildings:
                
                if len(X_train) == 0:
                    X_train = features
                else:
                    X_train = np.vstack((X_train, features))
                
                if len(Y_train) == 0:
                    Y_train = output
                else:
                    Y_train = np.vstack((Y_train, output))
            
            else:
                
                if len(X_test) == 0:
                    X_test = features
                else:
                    X_test = np.vstack((X_test, features))
                
                if len(Y_test) == 0:
                    Y_test = output
                else:
                    Y_test = np.vstack((Y_test, output))
    
    return (X_train, Y_train), (X_test, Y_test)        


def preprocess(X):
    """
    returns centered and normalized data
    """
    
    n, d = X.shape
    
    for i in range(d):
        
        mu = np.mean(X[:, i])
        var = np.var(X[:, i])
        
        if var == 0:
            var = 1
        
        X[:, i] = (X[:, i] - mu) / np.sqrt(var)
        
    return X