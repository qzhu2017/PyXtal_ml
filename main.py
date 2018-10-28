from pyxtal_ml.run import run
from pkg_resources import resource_filename

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
algos = ['Lasso']
N_sample = 100
feature = 'Chem'
level = 'tight'
pipeline = 'VT'

runner = run(N_sample=N_sample, jsonfile=jsonfile, level=level, feature=feature, pipeline = pipeline)
runner.load_data()
runner.convert_data_1D()
runner.choose_feature()
for algo in algos:
    runner.ml_train(algo=algo)
    #runner.print_outliers()
