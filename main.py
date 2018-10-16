import numpy as np
from descriptors.descriptors import descriptor
from datasets.collection import Collection
<<<<<<< HEAD
from ml.ML import *
=======
from ml.method import method
>>>>>>> 16037eac8747e1fd9739b2fadc379beda46269fa
from time import time
import warnings
warnings.filterwarnings("ignore")


<<<<<<< HEAD
file = 'datasets/sp_metal_aflow_844.json'
prop = 'form_energy_cell'
feature = 'RDF'  # 'RDF', 'RDF+ADF', 'all'
algorithm = 'KRR'
Kernel = 'laplacian'
=======
file = 'datasets/nonmetal_MP_8049.json'
prop = 'band_gap' #'formation_energy'
feature = 'RDF+Chem'  # 'RDF', 'RDF+ADF', 'all'
algo = 'KRR'
>>>>>>> 16037eac8747e1fd9739b2fadc379beda46269fa

# obtain the struc/prop data from source 
start = time()
strucs, props = Collection(file, prop).extract_struc_prop()
end = time()
print('Time elapsed for reading the json files: {:.3f} seconds'.format(end-start))
print('Total number of materials: {:d}'.format(len(strucs)))
print('The chosen feature is: {0}'.format(feature))

# convert the structures to descriptors in the format of 1D array
start = time()
x = []
for struc in strucs:
    #print(help(struc.to))
    struc.to(filename='1.vasp', fmt='poscar')
    des = descriptor(struc, feature).merge()
    if len(x) == 0:
        x = des
    else:
        x = np.vstack((x, des))

# Y has None values
y = np.array(props)
X = []
Y = []
for i in range(len(y)):
    if y[i] != None:
        Y.append(y[i])
        X.append(x[i])

end = time()
print('Time elapsed for creating the descriptors: {:.3f} seconds'.format(end-start))
print('Each material has {:d} descriptors'.format(np.shape(X)[1]))

<<<<<<< HEAD
# build machine learning model
ML = Machinelearning(algo = algorithm, feature = X, prop = Y)
r2, MAE = ML.KRR(Kernel)
print(r2,MAE)
=======
# build machine learning model for X/Y set
# to complete soon
start = time()
ml = method(X, Y, algo=algo)
end = time()
print('Time elapsed for machine learning: {:.3f} seconds'.format(end-start))
ml.plot()

>>>>>>> 16037eac8747e1fd9739b2fadc379beda46269fa
