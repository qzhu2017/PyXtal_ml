import yaml
import numpy as np

arr = np.logspace(-5,5)

default_params = {
        'KNN':
        {'light': {"n_neighbors": [5], "p": [2], "leaf_size": [30]},
            'medium': {"n_neighbors":[list(range(4,7))], "p":[1.0,2.0]},
            'tight': {"n_neighbors":[list(range(3,11))], "p":[0.5,1.0,1.5,2.0], "leaf_size":[10,30,60,100,150]}},
        'KRR':
        {'light': {"alpha": [1], "gamma": [1], "kernel": ['rbf']}, 
            'medium': {"alpha": [100, 10, 1, 0.1, 1e-2], "gamma": np.logspace(-2,2, 10), "kernel": ['rbf', 'laplacian']},
            'tight': {"alpha": [1e4, 1e3, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4], "gamma": np.logspace(-5,5), "kernel": ['rbf', 'laplacian', 'linear']}},
        'GradientBoosting': 
        {'light': {"GBR__learning_rate": [0.1], "GBR__n_estimators": [100], "fs__threshold": [0.0]}, 
            'medium': {"GBR__learning_rate": [0.1], "GBR__n_estimators": [100, 1500, 3000], "fs__threshold": [0.01]},
            'tight': {"GBR__learning_rate": [0.01, 0.1, 1, 10], "GBR__n_estimators": [100, 500, 1000, 1500, 2500, 3000, 4000, 5000], "fs__threshold": [0.0, 0.05, 0.1, 0.5]}},
        'RF':
        {'light': {"RFR__n_estimators": [10], "fs__threshold": [0.0]},
            'medium': {"RFR__n_estimators": [10,50,100,1000], "fs_threshold": [0.01]},
            'tight': {"RFR__n_estimators": [10, 30, 60, 90, 150, 250, 500, 750, 1000, 2000], "fs_threshold": [0.0, 0.05, 0.1, 0.5]}},
        'StochasticGD':
        {'light': {"penalty": ['l2'], "alpha": [0.1], "learning_rate": ['optimal']},
            'medium': {"penalty": ['l2'], "alpha": [100, 10, 1, 0.1, 1e-2], "learning_rate": ['optimal']},
            'tight': {"penalty": ['l2', 'elasticnet'], "alpha": [1e5, 1e4, 1e3, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5], "learning_rate": ['optimal', 'constant', 'invscaling', 'adaptive']}}
        }


with open('default_params.yml', 'w') as outfile:
    yaml.dump(ml_params, outfile, default_flow_style=False)

with open('default_params.yml', 'r') as stream:
    try:
        data = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
