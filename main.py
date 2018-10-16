import numpy as np
from descriptors.descriptors import descriptor
from datasets.collection import Collection
from ml.ML import *
from time import time
import warnings
warnings.filterwarnings("ignore")


file = 'datasets/sp_metal_aflow_844.json'
prop = 'form_energy_cell'
feature = 'RDF'  # 'RDF', 'RDF+ADF', 'all'
algorithm = 'KRR'
Kernel = 'laplacian'

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
#print(np.shape(Y))

# build machine learning model
ML = Machinelearning(algo = algorithm, feature = X, prop = Y)
r2, MAE = ML.KRR(Kernel)
print(r2,MAE)
