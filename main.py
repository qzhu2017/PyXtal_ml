from pyxtal_ml.run import run
from pkg_resources import resource_filename

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
algos = ['RF', 'GradientBoosting', 'KNN', 'KRR']
N_sample = 100
feature='Chem+RDF+ADF+Charge+Voronoi'
level = 'medium'

runner = run(N_sample=N_sample, jsonfile=jsonfile, level=level, feature=feature)
runner.load_data()
runner.convert_data_1D()
runner.choose_feature()
for algo in algos:
    runner.ml_train(algo=algo)
    #runner.print_outliers()
