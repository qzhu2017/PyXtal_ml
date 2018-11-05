from pyxtal_ml.run import run
from pkg_resources import resource_filename
import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
algos = ['GPR']
N_sample = 300
feature = 'Chem'
feature_scaling = 'MaxAbsScaler()'
level = 'tight' #{'my_params': {"kernel": [ExpSineSquared(l, p) for l in np.logspace(-2, 2, 10) for p in np.logspace(0, 2, 10)]}, 'CV': 4}
pipeline = False

runner = run(N_sample=N_sample, jsonfile=jsonfile, level=level, feature=feature, scale_feature = feature_scaling)
runner.load_data()
runner.convert_data_1D() #choose cpu number if you want to active this function
runner.choose_feature(keys='Chem') #choose feature combinations if you want
for algo in algos:
    runner.ml_train(algo=algo, pipeline=pipeline)
runner.print_time()
