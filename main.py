import numpy as np
import pandas as pd
import os.path as op
from tabulate import tabulate
from pyxtal_ml.descriptors.descriptors import descriptor
from pyxtal_ml.datasets.collection import Collection
from pyxtal_ml.ml.method import method
from time import time
import warnings
warnings.filterwarnings("ignore")

#file = 'datasets/nonmetal_MP_8049.json'
#prop = 'formation_energy' #'band_gap'
#feature = 'Chem+Charge'  # 'RDF', 'RDF+ADF', 'all'
#algo = 'KRR' #'GradientBoosting'
#parameters = 'light'
#figname = 'test_plot.png'
#N_sample = None #5000

file = op.join(op.dirname(__file__), 'pyxtal_ml/datasets/nonmetal_MP_8049.json')
prop = 'formation_energy' #'band_gap'
feature = 'RDF+ADF+Chem+Charge'  # 'RDF', 'RDF+ADF', 'all'
algo = 'KNN'
parameters = 'medium' #'medium' #'tight'
figname = 'test_plot.png'
N_sample = 200

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
    #struc.to(filename='1.vasp', fmt='poscar')
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
tag = {'prop': prop, 'feature':feature}
start = time()
ml = method(feature=X, prop=Y, algo=algo, params=parameters, tag=tag)
end = time()
print('Time elapsed for machine learning: {:.3f} seconds'.format(end-start))
ml.plot_correlation(figname=figname)
ml.print_summary()

# save cross-val results
#cv = pd.DataFrame.from_dict(ml.cv_result)
#cv.to_csv('results/CV_result.csv')

# print outliers
import collections
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
col_name = collections.OrderedDict(
                                   {'Formula': [],
                                   'Space group': [],
                                   'Nsites': [],
                                   'dY': [],
                                   }
                                  )
for id, diff in enumerate(ml.estimator.predict(X)-Y):
    if abs(diff) > 3*ml.mae:
        col_name['Formula'].append(strucs[id].composition.get_reduced_formula_and_factor()[0])
        col_name['Space group'].append(SpacegroupAnalyzer(strucs[id]).get_space_group_symbol())
        col_name['Nsites'].append(len(strucs[id].species))
        col_name['dY'].append(diff)

df = pd.DataFrame(col_name)
df = df.sort_values(['dY','Space group','Nsites'], ascending=[True, True, True])
print(tabulate(df, headers='keys', tablefmt='psql'))
