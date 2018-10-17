import numpy as np
from descriptors.descriptors import descriptor
from datasets.collection import Collection
from ml.method import method
from time import time
import warnings
warnings.filterwarnings("ignore")


file = 'datasets/nonmetal_MP_8049.json'
prop = 'band_gap' #'formation_energy'
feature = 'Chem'  # 'RDF', 'RDF+ADF', 'all'
algo = 'KRR'
figname = 'res.png'
N_sample = 100

# obtain the struc/prop data from source 
start = time()
strucs, props = Collection(file, prop, N_sample).extract_struc_prop()
end = time()
print('Time elapsed for reading the json files: {:.3f} seconds'.format(end-start))
print('Total number of materials: {:d}'.format(len(strucs)))
print('The chosen feature is: {0}'.format(feature))

# convert the structures to descriptors in the format of 1D array
start = time()
x = []
for i, struc in enumerate(strucs):
    #print(help(struc.to))
    struc.to(filename='1.vasp', fmt='poscar')
    des = descriptor(struc, feature).merge()
    if len(x) == 0:
        x = des
    else:
        x = np.vstack((x, des))
    if i%500 == 0:
        print('{} materials have been processed'.format(i))

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

# build machine learning model for X/Y set
# to complete soon
start = time()
ml = method(feature=X, prop=Y, algo=algo)
end = time()
print('Time elapsed for machine learning: {:.3f} seconds'.format(end-start))
ml.plot_correlation(figname=figname)
