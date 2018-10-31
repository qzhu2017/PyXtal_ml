"""
This example illustrate how to use pyxtal_ml to train the formation
energy of ~26000 materials from the Jarvis
It test the performance of different combination of featuress+algorithm.
RF and GradientBoosting are used since they appear to yield the best 
performance. An example results would look as follows

load_data                   8.89 seconds
convert_data             1312.70 seconds
choose_feature              0.43 seconds
ml                       1459.44 seconds

 Algo          features  length       mae        r2
   RF      Chem+Voronoi     484  0.135955  0.951396
   RF  Chem+ADF+Voronoi     574  0.136762  0.951857

The total run takes about ~1.5 hr for the entire calculation on 1 CPU.
If you just want to get some quick results, try to modify the parameter
of N_sample = 1000

"""
from pyxtal_ml.run import run
from pkg_resources import resource_filename
import os.path as op
import pandas as pd
import sys
import os

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("Summary.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()
pipeline = 'VT'
jsonfile = resource_filename("pyxtal_ml", "datasets/jarvis_25923.json")
level = {'params':{"n_estimators":[10, 50]}, 'CV':4} 
runner = run(jsonfile=jsonfile, N_sample=None, level=level, feature='Chem+ADF+Voronoi')
runner.load_data()
runner.convert_data_1D(parallel=16)

col = {'r2':[],
       'mae':[],
       'length':[],
       'features':[],
       'Algo':[],
      }

algos = ['RF']
tags = ['Chem+Voronoi', 
        'Chem+ADF+Voronoi', 
       ]

for keys in tags:
    for algo in algos:
        runner.choose_feature(keys=keys)
        runner.ml_train(algo=algo, save=True, pipeline=pipeline)
        runner.print_time()
        col['r2'].append(runner.ml.r2)
        col['mae'].append(runner.ml.mae)
        col['features'].append(keys)
        col['length'].append(len(runner.X[0]))
        col['Algo'].append(algo)
        filename1 = algo + '.joblib'
        filename2 = keys + '-' + algo + '.joblib'
        cmd = 'mv ' + filename1 + ' ' + filename2
        os.system(cmd)

df = pd.DataFrame(col)
print('\nThe summary of ml training with different descriptors\n')
print(df)
