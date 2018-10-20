import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import yaml
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

    def __init__(self, algo, feature, prop, tag, test_size = 0.3, **kwargs):
        """

        """
        self.algo = algo
        self.feature = feature
        self.prop = prop
        self.tag = tag
        self.test_size = test_size
        options = ['KNN', 'KRR', 'GradientBoosting', 'RF', 'StochasticGD']
        self.parameters_level = ['light', 'medium', 'tight']
        self.dict = kwargs
        
        if self.algo in options:
            # Split data into training and test sets
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.feature, self.prop, test_size = self.test_size, random_state = 0)
            
            with open('ml/default_params.yml', 'r') as stream:
                try:
                    self.ml_params = yaml.load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

        else:
            print('Warning: The Machine Learning algorithm is not available.')

        self.ml()

    def read_dict(self):
        """
        reading dictionary
        """
        for key, value in self.dict.items():
            if value in self.parameters_level:
                self.level = value # using light, medium, or tight
                break
            else:
                self.level = None # using user's defined parameters
                break

    def ml(self):
        """

        """
        if self.algo == 'KNN':
            grid = self.ml_params[self.algo]
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
            grid = self.ml_params[self.algo]
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
            grid = self.ml_params[self.algo]
            self.read_dict()
            
            if self.level == 'light':
                best_estimator = GradientBoostingRegressor()
            elif self.level == 'medium':
                best_estimator = GridSearchCV(GradientBoostingRegressor(), param_grid = {}, cv = 5, iid = False, return_train_score = False)
            else: # tight or user-defined parameters
                self.GB_grid = self.dict
                best_estimator = GridSearchCV(GradientBoostingRegressor(), param_grid = self.GB_grid, cv = 5, iid = False, return_train_score = False)

        best_estimator.fit(self.X_train, self.Y_train)
        self.y_predicted = best_estimator.predict(self.X_test)
        self.y_predicted0 = best_estimator.predict(self.X_train)
        self.r2 = best_estimator.score(self.X_test, self.Y_test, sample_weight=None)
        self.mae = mean_absolute_error(self.y_predicted, self.Y_test)

        if self.level == 'tight' or self.level == None:
            self.best_parameters = best_estimator.best_params_
        else:
            pass

    def plot_correlation(self, figname=None, figsize=(12,8)):
        """
        plot the correlation between prediction and target values
        """
        plt.figure(figsize=figsize)
        plt.scatter(self.y_predicted, self.Y_test, c='green', label='test')
        plt.scatter(self.y_predicted0, self.Y_train, c='blue', label='train')
        plt.title('{0:d} materials, r$^2$ = {1:.4f}, Algo: {2:s}'.format(len(self.prop), self.r2, self.algo))
        plt.xlabel('Prediction')
        plt.ylabel(self.tag['prop'])
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
        plt.figure(figsize=figsize)
        plt.hist(self.Y, bins = 100)
        plt.xlabel(self.tag['prop'])
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close()
       

    def print_summary(self):
        """
        print the paramters and performances
        """
        print("----------------------------------------")
        print("-------------SUMMARY of ML--------------")
        print("----------------------------------------")
        print("Number of samples:  {:20d}".format(len(self.prop)))
        print("Number of features: {:20d}".format(len(self.feature[0])))
        print("Algorithm:          {:>20}".format(self.algo))
        print("Feature:            {:>20}".format(self.tag['feature']))
        print("Property:           {:>20}".format(self.tag['prop']))
        print("r^2:              {:22.4f}".format(self.r2))
        print("MAE:              {:22.4f}".format(self.mae))

#        elif self.algo == 'RF':
#            grid = self.ml_params[self.algo]
#
#            self.read_dict()
#            if self.level in self.parameters_level:
#                self.RF_grid = grid[self.level]
#            else:
#                self.RF_grid = self.dict
#
#            RFR = RandomForestRegressor()
#            varthres = VarianceThreshold()
#            pipe = Pipeline([("fs", varthres),("RFR", RFR)])
#            search = GridSearchCV(pipe, self.RF_grid, cv=10,iid=False, return_train_score=False)
#            search.fit(self.X_train,self.Y_train)
#            
#            best_n_estimators = search.best_params_['RFR__n_estimators']
#            best_threshold = search.best_params_['fs__threshold']
#            
#            best_RFR = RandomForestRegressor(n_estimators = best_n_estimators)
#            best_estimator = Pipeline([("fs", VarianceThreshold(threshold = best_threshold)),("RFR", best_RFR)])
#
#        elif self.algo == 'StochasticGD':
#            grid = self.ml_params[self.algo]
#
#            self.read_dict()
#            if self.level in self.parameters_level:
#                self.SGD_grid = grid[self.level]
#            else:
#                self.SGD_grid = self.dict
#
#            search = GridSearchCV(SGDRegressor(tol = 1e-5), param_grid = self.SGD_grid, cv=10)
#            search.fit(self.X_train,self.Y_train)
#
#            best_penalty = search.best_params_['penalty']
#            best_alpha = search.best_params_['alpha']
#            best_learning_rate = search.best_params_['learning_rate']
#
#            best_estimator = SGDRegressor(tol = 1e-5, penalty = best_penalty, alpha = best_alpha, learning_rate = best_learning_rate)
