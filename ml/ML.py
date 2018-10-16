import sys
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error
from monty.json import MontyEncoder, MontyDecoder
from monty.serialization import loadfn, dumpfn
from pymatgen.core.structure import Structure
sys.path.append('../')
from descriptors.RDF import *

class Machinelearning:
    """

    """

    def __init__(self, algo, feature, prop):
        """

        """
        self.algo = algo
        self.feature = feature
        self.prop = prop
        options = ['KNN', 'KRR']

        if self.algo in options:
            # Split data into training and test sets
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.feature, self.prop, test_size = 0.2, random_state = 0)
            
        else:
            print('The Machine Learning algorithm is not available')

    def KRR(self, Kernel):
        """

        """
        p_grid = {"alpha": [1e3, 100, 10, 1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2,2)}
        estimator = GridSearchCV(KernelRidge(kernel = Kernel), cv=10, param_grid = p_grid)
        estimator.fit(self.X_train, self.Y_train)

        best_alpha = estimator.best_params_['alpha']
        best_gamma = estimator.best_params_['gamma']

        estimator2 = KernelRidge(alpha = best_alpha, gamma = best_gamma, kernel = Kernel, kernel_params = None)
        estimator2.fit(self.X_train, self.Y_train)
        y_pred = estimator2.predict(self.X_test)
        r2 = estimator2.score(self.X_test, self.Y_test, sample_weight=None)
        mae = mean_absolute_error(y_pred, self.Y_test)

        return r2, mae
   
