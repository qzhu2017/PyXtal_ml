import numpy as np
import collections
import pandas as pd
import os.path as op
from tabulate import tabulate
from time import time
from pyxtal_ml.descriptors.descriptors import descriptor
from pyxtal_ml.datasets.collection import Collection
from pyxtal_ml.ml.ml_sklearn import method
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from progressbar import ProgressBar
from optparse import OptionParser
import warnings
warnings.filterwarnings("ignore")

class run:
    """
    a class of production runs of pyxtal_ml
    Attributes:
        algo:
        time:
    """

    def __init__(self, jsonfile, N_sample=None, feature='Chem+RDF', 
                 prop='formation_energy', level='light'):
        """
        Args:
            algo: algorithm in ['KRR', 'KNN', ....]
            feature: features among ['Chem', 'RDF', ....]
            prop: target property in ['formation_energy', 'band_gap']
            level: 'light', 'medium', 'tight'
            file: source json file
        """
        self.feature0 = feature
        self.prop = prop
        self.level = level
        self.N_sample = N_sample
        self.file = jsonfile
        self.time = {}

    def load_data(self):
        """
        obtain the struc/prop data from source 
        """
        start = time()
        self.strucs, self.props = Collection(self.file, self.prop, self.N_sample).extract_struc_prop()
        end = time()
        self.time['load_data'] = end-start

    def convert_data_1D(self):
        """
        convert the structures to descriptors in the format of 1D array
        """
        start = time()
        print('Calculating the descriptors for {:} materials'.format(len(self.strucs)))
        pbar = ProgressBar()
        feas = []
        for struc in pbar(self.strucs):
            feas.append(descriptor(struc, self.feature0))
        end = time()
        self.time['convert_data'] = end-start

        self.features = feas

    def choose_feature(self, keys=None):
        """
        convert the structures to descriptors in the format of 1D array
        """
        X = []
        Y = []
        for fea, y0 in zip(self.features, self.props):
            if y0 != None:
                X.append(fea.merge(keys=keys))
                Y.append(y0)
        self.X = X
        self.Y = Y
        if keys is None:
            self.feature = self.feature0
        else:
            self.feature = keys

    def ml_train(self, algo='KRR', plot=False, print_info=True):
        """
        build machine learning model for X/Y set
        """
        self.algo = algo
        print('\nML learning with {} algorithm'.format(self.algo))
        tag = {'prop': self.prop, 'feature':self.feature}
        start = time()
        ml = method(feature=self.X, prop=self.Y, algo=self.algo, params=self.level, tag=tag)
        end = time()
        self.time['ml'] = end-start
        if plot:
            ml.plot_correlation(figname=self.file[:-4]+'_'+self.algo+'.png')
        if print_info:
            ml.print_summary()
        self.ml = ml

    def print_time(self):
        """
        print the timings
        """
        for tag in self.time.keys():
            print("{0:<16}  {1:>14.2f} seconds".format(tag, self.time[tag]))

    def print_outliers(self):
        """
        print the outlier information
        todo: make the output as an option
        """
        col_name = collections.OrderedDict(
                                   {'Formula': [],
                                   'Space group': [],
                                   'Nsites': [],
                                   'dY': [],
                                   }
                                  )
        for id, diff in enumerate(self.ml.estimator.predict(self.X)-self.Y):
            if abs(diff) > 3*self.ml.mae:
                struc = self.strucs[id]
                col_name['Formula'].append(struc.composition.get_reduced_formula_and_factor()[0])
                col_name['Space group'].append(SpacegroupAnalyzer(struc).get_space_group_symbol())
                col_name['Nsites'].append(len(struc.species))
                col_name['dY'].append(diff)
        
        df = pd.DataFrame(col_name)
        df = df.sort_values(['dY','Space group','Nsites'], ascending=[True, True, True])
        print('\nThe following structures have relatively high error compared to the reference values')
        print(tabulate(df, headers='keys', tablefmt='psql'))

if __name__ == "__main__":
    # -------------------------------- Options -------------------------
    parser = OptionParser()
    parser.add_option("-j", "--json", dest="jsonfile", default='',
                      help="json file, REQUIRED")
    parser.add_option("-a", "--algo", dest="algorithm", default='KRR',
                      help="algorithm, default: KRR")
    parser.add_option("-f", "--feature", dest="feature", default='Chem+RDF',
                      help="feature, default: Chem+RDF")
    parser.add_option("-p", "--prop", dest="property", default='formation_energy',
                      help="proerty, default: formation_energy")
    parser.add_option("-l", "--level", dest="level", default='medium',
                      help="level of fitting, default: medium")
    parser.add_option("-n", "--n_sample", dest="sample", default=200,
                      help="number of samples for ml, default: 200")


    (options, args) = parser.parse_args()
    print(options.jsonfile)
    runner = run(algo=options.algorithm, feature=options.feature, prop=options.property,
                 level=options.level, N_sample=options.sample, jsonfile=options.jsonfile)
    runner.load_data()
    runner.convert_data_1D()
    runner.ml_train()
    runner.print_outliers()


