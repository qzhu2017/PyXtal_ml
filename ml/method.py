import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_absolute_error
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams.update({'figure.autolayout': True})
plt.style.use("bmh")
sys.path.append('../')

class method:
    """
    the class of ml model training
    """

    def __init__(self, algo, feature, prop, test_size = 0.3, **kwargs):
        """

        """
        self.algo = algo
        self.feature = feature
        self.prop = prop
        self.test_size = test_size
        options = ['KNN', 'KRR', 'GradientBoosting', 'RF', 'StochasticGD']
        self.parameters_level = ['light', 'medium', 'tight']
        self.dict = kwargs
        
        if self.algo in options:
            # Split data into training and test sets
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.feature, self.prop, test_size = self.test_size, random_state = 0)
            
        else:
            print('Warning: The Machine Learning algorithm is not available.')

        self.ml()

    def read_dict(self):
        """
        reading dictionary
        """
        for key, value in self.dict.items():
            if value in self.parameters_level:
                self.level = value # using light medium tight
                break
            else:
                self.level = None # using user's defined parameters
                break

    def ml(self):
        """

        """
        if self.algo == 'KNN':
            grid = {'light': {"n_neighbors": [5], "p": [2], "leaf_size": [30]}, 
                    'medium': {"n_neighbors":[list(range(4,7))], "p":[1.0,2.0]},
                    'tight': {"n_neighbors":[list(range(3,11))], "p":[0.5,1.0,1.5,2.0], "leaf_size":[10,30,60,100,150]}}
            self.read_dict()
            if self.level in self.parameters_level:
                self.KNN_grid = grid[self.level]
            else:
                self.KNN_grid = self.dict

            search = GridSearchCV(KNeighborsRegressor(weights='distance'), cv=10, param_grid=self.KNN_grid)
            search.fit(self.X_train, self.Y_train)
            
            best_n_neighbors = search.best_params_['n_neighbors']
            best_p = search.best_params_['p']
            best_leaf_size = search.best_params_['leaf_size']
            
            best_estimator = KNeighborsRegressor(best_n_neighbors, weights='distance', algorithm='auto',leaf_size=best_leaf_size, p=best_p)
            
        elif self.algo == 'KRR':
            grid = {'light': {"alpha": [0.1], "gamma": [1], "kernel": ['rbf']}, 
                    'medium': {"alpha": [100, 10, 1, 0.1, 1e-2], "gamma": np.logspace(-2,2), "kernel": ['rbf', 'laplacian']},
                    'tight': {"alpha": [1e4, 1e3, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4], "gamma": np.logspace(-5,5), "kernel": ['rbf', 'laplacian', 'linear']}}
            self.read_dict()
            if self.level in self.parameters_level:
                self.KRR_grid = grid[self.level]
            else:
                self.KRR_grid = self.dict

            search = GridSearchCV(KernelRidge(), cv=10, param_grid = self.KRR_grid)
            search.fit(self.X_train, self.Y_train)
            
            best_alpha = search.best_params_['alpha']
            best_gamma = search.best_params_['gamma']
            best_kernel = search.best_params_['kernel']
            
            best_estimator = KernelRidge(alpha = best_alpha, gamma = best_gamma, kernel = best_kernel, kernel_params = None)
            
        elif self.algo == 'GradientBoosting':
            grid = {'light': {"GBR__learning_rate": [0.1], "GBR__n_estimators": [1000], "fs__threshold": [0.0]}, 
                    'medium': {"GBR__learning_rate": [0.1], "GBR__n_estimators": [100, 1500, 3000], "fs__threshold": [0.0]},
                    'tight': {"GBR__learning_rate": [0.01, 0.1, 1, 10], "GBR__n_estimators": [100, 500, 1000, 1500, 2500, 3000, 4000, 5000], "fs__threshold": [0.0, 0.05, 0.1, 0.5]}}
            self.read_dict()
            if self.level in self.parameters_level:
                self.GB_grid = grid[self.level]
            else:
                self.GB_grid = self.dict

            GBR = GradientBoostingRegressor(loss = 'huber')
            varthres = VarianceThreshold()
            pipe = Pipeline([("fs", varthres),("GBR", GBR)])
            search = GridSearchCV(pipe, self.GB_grid, cv=10,iid=False, return_train_score=False)
            search.fit(self.X_train,self.Y_train)
            
            best_learning = search.best_params_['GBR__learning_rate']
            best_estimators = search.best_params_['GBR__n_estimators']
            best_threshold = search.best_params_['fs__threshold']
            
            best_GBR = GradientBoostingRegressor(loss='huber', learning_rate = best_learning, n_estimators = best_estimators)
            best_estimator = Pipeline([("fs", VarianceThreshold(threshold = best_threshold)),("GBR", best_GBR)])
        
        elif self.algo == 'RF':
            grid = {'light': {"RFR__n_estimators": [10], "fs__threshold": [0.01]},
                    'medium': {"RFR__n_estimators": [10,50,100,1000], "fs_threshold": [0.01]},
                    'tight': {"RFR__n_estimators": [10, 30, 60, 90, 150, 250, 500, 750, 1000, 2000], "fs_threshold": [0.0, 0.05, 0.1, 0.5]}}
            self.read_dict()
            if self.level in self.parameters_level:
                self.RF_grid = grid[self.level]
            else:
                self.RF_grid = self.dict

            RFR = RandomForestRegressor()
            varthres = VarianceThreshold()
            pipe = Pipeline([("fs", varthres),("RFR", RFR)])
            search = GridSearchCV(pipe, self.RF_grid, cv=10,iid=False, return_train_score=False)
            search.fit(self.X_train,self.Y_train)
            
            best_n_estimators = search.best_params_['RFR__n_estimators']
            best_threshold = search.best_params_['fs__threshold']
            
            best_RFR = RandomForestRegressor(n_estimators = best_n_estimators)
            best_estimator = Pipeline([("fs", VarianceThreshold(threshold = best_threshold)),("RFR", best_RFR)])

        elif self.algo == 'StochasticGD':
            grid = {'light': {"penalty": ['l2'], "alpha": [0.1], "learning_rate": ['optimal']},
                    'medium': {"penalty": ['l2'], "alpha": [100, 10, 1, 0.1, 1e-2], "learning_rate": ['optimal']},
                    'tight': {"penalty": ['l2', 'elasticnet'], "alpha": [1e5, 1e4, 1e3, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5], "learning_rate": ['optimal', 'constant', 'invscaling', 'adaptive']}}
            self.read_dict()
            if self.level in self.parameters_level:
                self.SGD_grid = grid[self.level]
            else:
                self.SGD_grid = self.dict

            search = GridSearchCV(SGDRegressor(tol = 1e-5), param_grid = self.SGD_grid, cv=10)
            search.fit(self.X_train,self.Y_train)

            best_penalty = search.best_params_['penalty']
            best_alpha = search.best_params_['alpha']
            best_learning_rate = search.best_params_['learning_rate']

            best_estimator = SGDRegressor(tol = 1e-5, penalty = best_penalty, alpha = best_alpha, learning_rate = best_learning_rate)

        best_estimator.fit(self.X_train, self.Y_train)
        self.y_predicted = best_estimator.predict(self.X_test)
        self.y_predicted0 = best_estimator.predict(self.X_train)
        self.r2 = best_estimator.score(self.X_test, self.Y_test, sample_weight=None)
        self.mae = mean_absolute_error(self.y_predicted, self.Y_test)

    def plot_correlation(self, figname=None, figsize=(12,8)):
        """
        plot the correlation between prediction and target values
        """
        plt.figure(figsize=figsize)
        plt.scatter(self.y_predicted, self.Y_test, c='green', label='test')
        plt.scatter(self.y_predicted0, self.Y_train, c='blue', label='train')
        plt.title('{0:d} materials, r$^2$ = {1:.4f}, Algo: {2:s}'.format(len(self.prop), self.r2, self.algo))
        plt.xlabel('Prediction')
        plt.ylabel('Reference')
        plt.legend()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close()
            
    def plot_distribution(self, figname=None, figsize=(12,8)):
        """
        some other plots to facilate the results
        """

    def print_summary(self):
        """
        print the paramters and performances
        """
        print("----------------------------------------")
        print("-------------SUMMARY of ML--------------")
        print("----------------------------------------")
        print("Number of samples:  {:20d}".format(len(self.prop)))
        print("Number of features: {:20d}".format(len(self.feature[0])))
        print("Algorithm:          {:^25}".format(self.algo))
        print("r^2:              {:22.4f}".format(self.r2))
        print("MAE:              {:22.4f}".format(self.mae))
