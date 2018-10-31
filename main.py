from pyxtal_ml.run import run
from pkg_resources import resource_filename

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
algos = ['RF']
N_sample = 100
feature = 'Chem'
level = {'my_params': {"n_estimators": [10]}, 'CV': 4}
pipeline = False

runner = run(N_sample=N_sample, jsonfile=jsonfile, level=level, feature=feature)
runner.load_data()
runner.convert_data_1D(parallel=2) #choose cpu number if you want to active this function
runner.choose_feature(keys='Chem') #choose feature combinations if you want
for algo in algos:
    runner.ml_train(algo=algo, pipeline=pipeline)
runner.print_time()
