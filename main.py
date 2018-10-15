import numpy as np
from descriptors.descriptors import descriptor
from datasets.collection import Collection
#from ml.method import method
from time import time
import warnings
warnings.filterwarnings("ignore")


file = 'datasets/sp_metal_aflow_844.json'
prop = 'form_energy_cell'
feature = 'all'  # 'RDF', 'RDF+ADF', 'all'

# obtain the struc/prop data from source 
start = time()
strucs, props = Collection(file, prop).extract_struc_prop()
end = time()
print('Time elapsed for reading the json files: {:.3f} seconds'.format(end-start))
print('Total number of materials: {:d}'.format(len(strucs)))
print('The chosen feature is: {0}'.format(feature))

# convert the structures to descriptors in the format of 1D array
start = time()
X = []
for struc in strucs:
    des = descriptor(struc, feature).merge()
    if len(X) == 0:
        X = des
    else:
        X = np.vstack((X, des))

Y = np.array(props)
end = time()
print('Time elapsed for creating the descriptors: {:.3f} seconds'.format(end-start))
print('Each material has {:d} descriptors'.format(np.shape(X)[1]))
#print(np.shape(Y))

# build machine learning model

