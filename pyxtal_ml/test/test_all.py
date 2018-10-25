from pyxtal_ml.run import run
from pkg_resources import resource_filename
import os.path as op

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")


runner = run(N_sample=200, jsonfile=jsonfile)
runner.load_data()
runner.convert_data_1D()
runner.ml_train()
runner.print_outliers()

