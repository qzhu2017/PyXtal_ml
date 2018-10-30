import yaml
import numpy as np

default_params = {
        'KNN':
        {'cv': 10,
            'params': {"n_neighbors":[3,4], "p":[1.0, 2.0], "leaf_size":[30,100]}},
        'KRR':
        {'cv': 10,
            'params': {"alpha": [1e3, 100, 10, 1, 0.1, 1e-2, 1e-3], "gamma": [-5, 1, 5], "kernel": ['rbf', 'laplacian', 'linear']}},
        'GradientBoosting': 
        {'cv': 10,
            'params': {"learning_rate": [0.01, 0.1, 1, 10], "n_estimators": [100, 500, 1000, 1500, 2500, 3000, 4000, 5000]}},
        'RF':
        {'cv': 10,
            'params': {"n_estimators": [10, 30, 60, 90, 150, 250]}},
        'StochasticGD':
        {'cv': 10,
            'params': {"penalty": ['l2', 'elasticnet'], "alpha": [1e5, 1e4, 1e3, 100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5], "learning_rate": ['optimal', 'constant', 'invscaling', 'adaptive']}},
	    'ANN':
        {'cv': 10,
            'params': {"n_estimators": [10, 30, 60, 90, 150, 250, 500, 750, 1000, 2000]}},
        'SVR':
        {'cv': 10,
            'params': {"gamma": [0.01, 0.1, 1, 10, 100], "epsilon": [1e-2, 1e-1, 1, 1e1, 1e2], "C": [1, 10, 100, 1000, 10000]}},
        'Lasso':
        {'cv': 10,
            'params': {"alpha": [1e3, 100, 10, 1, 0.1, 1e-2, 1e-3]}},
        'ENet':
        {'cv': 10,
            'params': {"alpha": [1e3, 100, 10, 1, 0.1, 1e-2, 1e-3], "l1_ratio": [1, 0.5, 0]}}
        }


with open('default_params.yaml', 'w') as outfile:
    yaml.dump(default_params, outfile, default_flow_style=False)

with open('default_params.yaml', 'r') as stream:
    try:
        data = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
