import sys
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use("bmh")
from pymatgen.core.structure import Structure
sys.path.append('../')
from descriptors.RDF import *

class method:
    """

    """

    def __init__(self, algo = 'KRR', feature, prop, test_size = 0.3):
        """

        """
        self.algo = algo
        self.feature = feature
        self.prop = prop
        self.test_size = test_size
        options = ['KNN', 'KRR', 'GradientBoosting']

        if self.algo in options:
            # Split data into training and test sets
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.feature, self.prop, test_size = self.test_size, random_state = 0)
            
        else:
            print('Warning: The Machine Learning algorithm is not available.')

    def ml(self):
        """

        """
        if self.algo == 'KNN':
            self.KNN_grid = {"n_neighbors": [list(range(2,10))], "p": [1,2], "leaf_size": [10,30,50,100]}
            search = GridSearchCV(KNeighborsRegressor(weights='distance', algorith='kd_tree'), cv=10, param_grid=self.KNN_grid)
            search.fit(self.X_train, self.Y_train)
            
            best_n_neighbors = search.best_params_['n_neighbors']
            best_p = search.best_params_['p']
            best_leaf_size = search.best_params_['leaf_size']
            
            best_estimator = KNeighborsRegressor(best_n_neighbors, weights='distance', algorithm='kd_tree',leaf_size=best_leaf_size, p=best_p)
            
        elif self.algo == 'KRR':
            self.KRR_grid = {"alpha": [1e3, 100, 10, 1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2,2)}
            search = GridSearchCV(KernelRidge(kernel = 'rbf'), cv=10, param_grid = self.KRR_grid)
            search.fit(self.X_train, self.Y_train)
            
            best_alpha = estimator.best_params_['alpha']
            best_gamma = estimator.best_params_['gamma']
            
            best_estimator = KernelRidge(alpha = best_alpha, gamma = best_gamma, kernel = 'rbf', kernel_params = None)
            
        elif self.algo == 'GradientBoosting':
            self.GB_grid = {'est__learning_rate': [0.1], 'est__n_estimators': [2000,3000]}
            est= GradientBoostingRegressor(loss = 'huber')
            varthres = VarianceThreshold(0.01)
            pipe = Pipeline([("fs", varthres),("est", est)])
            search = GridSearchCV(pipe, self.GB_grid, cv=10,iid=False, return_train_score=False)
            search.fit(X_train,Y_train)
            
            best_learning = search.best_params_['est__learning_rate']
            best_estimators = search.best_params_['est__n_estimators']
            
            best_est = GradientBoostingRegressor(loss='huber', learning_rate = best_learning, n_estimators = best_estimators)
            best_estimator = Pipeline([("fs", VarianceThreshold(threshold = 0.01)),("est", best_est)])
        
        best_estimator.fit(X_train, Y_train)
        self.y_predicted = best_estimator.predict(X_test)
        self.y_predicted0 = best_estimator.predict(X_train)
        self.r2 = best_estimator.score(X_test, Y_test, sample_weight=None)
        self.mae = mean_absolute_error(self.y_predicted, self.Y_test)

        return self.r2, self.mae
    
def plot(self, figname=None, figsize=(12,8)):
    """
    
    """
    plt.figure(figsize=figsize)
    plt.scatter(self.y_predicted, self.Y_test, c='green', label='test')
    plt.scatter(self.y_predicted0, self.Y_train, c='blue', label='train')
    plt.title('{0:d} materials, r$^2$ = {1:.4f}, Algo: {2:s}'.format(len(self.Y), self.r2, self.algo))
    plt.xlabel('Prediction')
    plt.ylabel('Reference')
    plt.legend()
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
