import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor, ElasticNet, Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import PCA
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
    Class for implementing a machine learning algorithm based on the comprehensiveness level
    of training. All machine learning algorithms is called from Scikit-learn. The minimum inputs 
    to employ this class are type of algorithm, feature as the descriptors, property to be predicted,
    and the tag consists of the names of features and properties.

    Args:
        algo: A string consists of machine learning algorithm defined in algo_options
        feature: A list of materials' feature
        prop: An array of materials' property
        tag: A dict of property and features names
        pipeline: Add extra machine learning algorithms to be run one after another 
        test_size: a default argument of 0.3 means 30% of data is used for testing 
            the machine learning model.
        kwargs: A dictionary of dictionaries of machine learning parameters.
    """

    def __init__(self, algo, feature, prop, tag, pipeline = False, test_size = 0.3, **kwargs):
        """

        """
        self.algo = algo
        self.feature = feature
        self.prop = prop
        self.tag = tag
        self.pipeline = pipeline
        self.test_size = test_size
        self.dict = kwargs
        self.algo_options = ['KNN', 'KneighborsRegressor', 'KRR', 'KernelRidge', 'GB', 'GradientBoosting', 
                'RF', 'RandomForestRegressor', 'SGD', 'SGDRegressor', 'MLPRegressor', 'ANN', 'SVR', 
                'Lasso', 'ElasticNet', 'ENet', 'GaussianProcessRegressor', 'GPR']
        self.pipeline_options = ['VT', 'VarianceThreshold', 'PCA']
        self.parameters_level = ['light', 'medium', 'tight']
        
        if self.algo in self.algo_options:
            # Split data into training and test sets
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.feature, self.prop, test_size = self.test_size, random_state = 0)
            
            with open(yaml_path, 'r') as stream:
                try:
                    self.algos_params = yaml.load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

        else:
            print('Warning: The Machine Learning algorithm is not available.')

        self.ml()

    def read_dict(self):
        """
        reading from **kwargs argument to determine the comprehensiveness level of training and its parameters
        """
        for key, value in self.dict.items():
            if value in self.parameters_level:
                self.level = value                          # light, medium, or tight
                self.params = self.algos_params[self.algo]  # import ml parameters pre-defined in default_params.yaml
                break
            else:
                self.level = None
                self.params = value                     # using user-defined parameters
                break
        
    def get_params_for_gridsearch(self, level, params_):
        """
        get parameters for GridSearch
        """
        keys = []
        for key, value in params_.items():
            keys.append(key)
            if type(value) is int:
                CV = value
            else:
                p_grid = value
                CV = 10
        if level == 'light':
            p_grid = {}
            CV = 2
        elif level == 'medium':
            p_grid = {}
        else:                                            # This should work for 'tight' and user-defined parameters
            pass
        
        return p_grid, CV

    def get_pipe_params_for_gridsearch(self, algo, grid):
        """
        get pipeline parameters for GridSearch
        """
        keys = []
        clf_grid = {}
        if len(grid) == 0:
            clf_grid = {}
        else:
            for key, value in grid.items():
                clf_grid[algo+'__'+key] = value

        return clf_grid

    def ml(self):
        """

        """
        self.read_dict()

        # Main classifier
        if self.algo in ['KNN', 'KNeighborsRegressor']:
            self.grid, self.CV = self.get_params_for_gridsearch(self.level, self.params)
            clf = KNeighborsRegressor()
           
        elif self.algo in ['KRR', 'KernelRidge']:
            self.grid, self.CV = self.get_params_for_gridsearch(self.level, self.params)
            clf = KernelRidge()

        elif self.algo in ['GB', 'GradientBoostingRegressor']:
            self.grid, self.CV = self.get_params_for_gridsearch(self.level, self.params)
            clf = GradientBoostingRegressor()
        
        elif self.algo in ['RandomForestRegressor', 'RF']:
            self.grid, self.CV = self.get_params_for_gridsearch(self.level, self.params)
            clf = RandomForestRegressor()

        elif self.algo in ['SGDRegressor', 'SGD']:
            self.grid, self.CV = self.get_params_for_gridsearch(self.level, self.params)
            clf = SGDRegressor()
        
        elif self.algo in ['SVR']:
            self.grid, self.CV = self.get_params_for_gridsearch(self.level, self.params)
            clf = SVR()

        elif self.algo in ['ElasticNet', 'ENet']:
            self.grid, self.CV = self.get_params_for_gridsearch(self.level, self.params)
            clf = ElasticNet()

        elif self.algo in ['Lasso']:
            self.grid, self.CV = self.get_params_for_gridsearch(self.level, self.params)
            clf = Lasso()
        
        # Extra classifier
        if self.pipeline in self.pipeline_options:
            self.grid = self.get_pipe_params_for_gridsearch(self.algo, self.grid)
            if self.pipeline in ['VarianceThreshold', 'VT']:
                estimator = Pipeline(steps=[(self.pipeline, VarianceThreshold()),(self.algo, clf)])
            elif self.pipeline == 'PCA':
                estimator = Pipeline(steps=[(self.pipeline, PCA()), (self.algo, clf)])
        else:
            estimator = clf

        best_estimator = GridSearchCV(estimator, param_grid = self.grid, cv = self.CV)
        best_estimator.fit(self.X_train, self.Y_train)
        self.y_predicted = best_estimator.predict(self.X_test)
        self.y_predicted0 = best_estimator.predict(self.X_train)
        self.r2 = r2_score(self.Y_test, self.y_predicted, sample_weight=None)
        self.mae = mean_absolute_error(self.y_predicted, self.Y_test)
        self.estimator = best_estimator
        self.cv_result = best_estimator.cv_results_

        if self.level == 'tight' or self.level == None:
            self.best_parameters = best_estimator.best_params_

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
        for score, std, param in zip(self.cv_result['mean_train_score'], self.cv_result['std_train_score'], self.cv_result['params']):
            print("Parameters: {}".format(param))
            print("Mean train_score: {:22.4f}".format(score))
            print("Std train_score:  {:22.4f}".format(std))
