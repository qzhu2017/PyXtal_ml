"""
This example illustrate how to use pyxtal_ml to train the formation
energy of 8049 materials from the Materials Project.
It test the performance of different combination of featuress+algorithm.
RF and GradientBoosting are used since they appear to yield the best 
performance. An example results would look as follows

load_data                   2.97 seconds
convert_data             3546.69 seconds
ml                         41.05 seconds

The summary of ml training with different descriptors

            Algo             features  length       mae        r2
GradientBoosting                 Chem     438  0.215161  0.914494
              RF                 Chem     438  0.198129  0.915552
GradientBoosting                  RDF      60  0.659517  0.394800
              RF                  RDF      60  0.629285  0.407018
GradientBoosting                  ADF      90  0.646748  0.422473
              RF                  ADF      90  0.569808  0.494114
GradientBoosting               Charge     756  0.517367  0.588184
              RF               Charge     756  0.412319  0.672367
GradientBoosting              Voronoi      39  0.336228  0.815856
              RF              Voronoi      39  0.279356  0.841874
GradientBoosting         Chem+Voronoi     477  0.202426  0.926171
              RF         Chem+Voronoi     477  0.170225  0.933439
GradientBoosting          Chem+Charge    1194  0.218181  0.913095
              RF          Chem+Charge    1194  0.198489  0.915879
GradientBoosting       Voronoi+Charge     795  0.332156  0.822286
              RF       Voronoi+Charge     795  0.257998  0.861395
GradientBoosting             Chem+RDF     498  0.219850  0.914296
              RF             Chem+RDF     498  0.195736  0.918474
GradientBoosting             Chem+ADF     528  0.216785  0.916249
              RF             Chem+ADF     528  0.191048  0.920803
GradientBoosting  Chem+RDF+ADF+Charge    1344  0.218465  0.915107
              RF  Chem+RDF+ADF+Charge    1344  0.196320  0.919079

The total run takes about ~1 hr for the entire calculation on 1 CPU.
If you just want to get some quick results, try to modify the parameter
of N_sample = 1000

"""
from pyxtal_ml.run import run
from pkg_resources import resource_filename
import os.path as op
import pandas as pd

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
runner = run(jsonfile=jsonfile, N_sample=None, feature='Chem+RDF+ADF+Charge+Voronoi')
runner.load_data()
runner.convert_data_1D()

col = {'r2':[],
       'mae':[],
       'length':[],
       'features':[],
       'Algo':[],
      }

algos = ['GradientBoosting', 'RF']
tags = ['Chem', 'RDF', 'ADF', 'Charge', 'Voronoi',
        'Chem+Voronoi', 'Chem+Charge', 'Voronoi+Charge',
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
