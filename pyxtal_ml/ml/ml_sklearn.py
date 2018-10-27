import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, ElasticNet, Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import yaml
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt
import os.path as op

rcParams.update({'figure.autolayout': True})
plt.style.use("bmh")
sys.path.append('../')
yaml_path = op.join(op.dirname(__file__), 'default_params.yaml')

class method:
    """
    Class for implementing a machine learning algorithm based on the level of comprehensiveness
    of training. The minimum inputs to employ this class are type of algorithm, feature as the 
    descriptors, property to be predicted
    """

    def __init__(self, algo, feature, prop, tag, pipeline = False, test_size = 0.3, **kwargs):
        """

        """
        self.algo = algo
        self.feature = feature
        self.prop = prop
        self.tag = tag
        self.test_size = test_size
        self.pipeline = pipeline
        self.ml_options = ['KNN', 'KRR', 'GradientBoosting', 'RF', 'StochasticGD', 'ANN', 'SVR', 'Lasso', 'ENet']
        self.parameters_level = ['light', 'medium', 'tight']
        self.dict = kwargs
        
        if self.algo in self.ml_options:
            # Split data into training and test sets
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.feature, self.prop, test_size = self.test_size, random_state = 0)
            
            with open(yaml_path, 'r') as stream:
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
                self.level = value # light, medium, or tight
                self.params_ = self.ml_params[self.algo] # import ml parameters defined in default_params.yaml
                break
            else:
                self.level = None
                self.params_ = self.dict # using user's defined parameters
                break
        
    def gridsearch_params(self, level, dict_params_):
        """
        


        """
        keys = []
        for key, value in dict_params_.items():
            keys.append(key)
        if level == 'light':
            p_grid = {}
            CV = 2
        elif level == 'medium':
            p_grid = {}
            CV = dict_params_[keys[0]]
        else: # This should work for 'tight' and user-defined parameters
            p_grid = dict_params_[keys[1]]
            CV = dict_params_[keys[0]]

        return p_grid, CV

    def ml(self):
        """

        """
        self.read_dict()

        if self.algo == 'KNN':

            self.KNN_grid, self.CV = self.gridsearch_params(self.level, self.ml_params[self.algo])
            best_estimator = GridSearchCV(KNeighborsRegressor(), param_grid = self.KNN_grid, cv = self.CV)
           
        elif self.algo == 'KRR':

            self.KRR_grid, self.CV = self.gridsearch_params(self.level, self.ml_params[self.algo])
            best_estimator = GridSearchCV(KernelRidge(), param_grid = self.KRR_grid, cv = self.CV)

        elif self.algo == 'GradientBoosting':

            self.GB_grid, self.CV = self.gridsearch_params(self.level, self.ml_params[self.algo])
            best_estimator = GridSearchCV(GradientBoostingRegressor(), param_grid = self.GB_grid, cv = self.CV)

        elif self.algo == 'RF':

            self.RF_grid, self.CV = self.gridsearch_params(self.level, self.ml_params[self.algo])
            best_estimator = GridSearchCV(RandomForestRegressor(), param_grid = self.RF_grid, cv = self.CV)

        elif self.algo == 'StochasticGD':

            self.SGD_grid, self.CV = self.gridsearch_params(self.level, self.ml_params[self.algo])
            best_estimator = GridSearchCV(SGDRegressor(), param_grid = self.SGD_grid, cv = self.CV)
        
        elif self.algo == 'SVR':

            self.SVR_grid, self.CV = self.gridsearch_params(self.level, self.ml_params[self.algo])
            best_estimator = GridSearchCV(SVR(), param_grid = self.SVR_grid, cv = self.CV)

        elif self.algo == 'ENet':

            self.ENet_grid, self.CV = self.gridsearch_params(self.level, self.ml_params[self.algo])
            best_estimator = GridSearchCV(ElasticNet(), param_grid = self.ENet_grid, cv = self.CV)

        elif self.algo == 'Lasso':

            self.Lasso_grid, self.CV = self.gridsearch_params(self.level, self.ml_params[self.algo])
            best_estimator = GridSearchCV(Lasso(), param_grid = self.Lasso_grid, cv = self.CV)
        
        if self.pipeline == False:
            best_estimator.fit(self.X_train, self.Y_train)
            self.y_predicted = best_estimator.predict(self.X_test)
            self.y_predicted0 = best_estimator.predict(self.X_train)
            self.r2 = r2_score(self.Y_test, self.y_predicted, sample_weight=None)
            self.mae = mean_absolute_error(self.y_predicted, self.Y_test)
            self.estimator = best_estimator
        
            if self.level in ['tight', 'medium']:
                self.cv_result = best_estimator.cv_results_

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
        if self.level in ['tight', 'medium']:
            for score, std in zip(self.cv_result['mean_train_score'], self.cv_result['std_train_score']):
                print("Mean train_score: {:22.4f}".format(score))
                print("Std train_score:  {:22.4f}".format(std))
