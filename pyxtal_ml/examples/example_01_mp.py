"""
This example illustrate how to use pyxtal_ml to train the formation
energy of 8049 materials from the Materials Project.
It test the performance of different combination of featuress+algorithm.
An example results would look as follows

The summary of ml training with different descriptors

             Algo             features  length       mae        r2
              KNN                 Chem     438  0.631907  0.411481
              KRR                 Chem     438  0.382373  0.706129
 GradientBoosting                 Chem     438  0.219900  0.915303
               RF                 Chem     438  0.197587  0.916510
              KNN          Chem+Charge    1194  0.631951  0.411270
              KRR          Chem+Charge    1194  0.380000  0.750524
 GradientBoosting          Chem+Charge    1194  0.223244  0.912955
               RF          Chem+Charge    1194  0.198653  0.917440
              KNN             Chem+RDF     498  0.629574  0.413876
              KRR             Chem+RDF     498  0.601662 -0.188306
 GradientBoosting             Chem+RDF     498  0.221232  0.915225
               RF             Chem+RDF     498  0.194533  0.919611
              KNN             Chem+ADF     528  0.630569  0.411797
              KRR             Chem+ADF     528  0.383746  0.715839
 GradientBoosting             Chem+ADF     528  0.221605  0.915490
               RF             Chem+ADF     528  0.189786  0.921913
              KNN  Chem+RDF+ADF+Charge    1344  0.629519  0.413704
              KRR  Chem+RDF+ADF+Charge    1344  0.418701  0.605264
 GradientBoosting  Chem+RDF+ADF+Charge    1344  0.221663  0.915792
               RF  Chem+RDF+ADF+Charge    1344  0.198125  0.919601

The total run takes about ~ mins for the entire calculation on 1 CPU.
If you just want to get some quick results, try to modify the parameter
of N_sample = 1000

"""
from pyxtal_ml.run import run
from pkg_resources import resource_filename
import os.path as op
import pandas as pd

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
runner = run(jsonfile=jsonfile, N_sample=None, feature='Chem+RDF+ADF+Charge')
runner.load_data()
runner.convert_data_1D()

col = {'r2':[],
       'mae':[],
       'length':[],
       'features':[],
       'Algo':[],
      }

algos = ['KNN', 'KRR', 'GradientBoosting', 'RF']
tags = ['Chem', 'RDF', 'ADF', 'Charge', 'Chem+Charge', 
        'Chem+RDF', 'Chem+ADF', 'Chem+RDF+ADF+Charge']
for keys in tags:
    for algo in algos:
        runner.choose_feature(keys=keys)
        runner.ml_train(algo=algo)
        runner.print_time()
        col['r2'].append(runner.ml.r2)
        col['mae'].append(runner.ml.mae)
        col['features'].append(keys)
        col['length'].append(len(runner.X[0]))
        col['Algo'].append(algo)

df = pd.DataFrame(col)
print('\nThe summary of ml training with different descriptors\n')
print(df)
