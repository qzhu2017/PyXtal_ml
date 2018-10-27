from pyxtal_ml.run import run
from pkg_resources import resource_filename
from sklearn.externals import joblib
import numpy as np

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
algos = ['RF', 'GradientBoosting', 'KNN', 'KRR']
N_sample = 100
feature='Chem+RDF+ADF+Charge+Voronoi'

runner = run(N_sample=N_sample, jsonfile=jsonfile, feature=feature)
runner.load_data()
runner.convert_data_1D()
runner.choose_feature()
model = joblib.load('RF.joblib')
diff = model.predict(runner.X) - runner.Y
print(np.mean(diff), np.std(diff))


