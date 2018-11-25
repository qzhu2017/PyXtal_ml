import numpy as np
import sys
import collections
import pandas as pd
import os.path as op
from tabulate import tabulate
from time import time
from sklearn.preprocessing import (MinMaxScaler, minmax_scale, MaxAbsScaler, maxabs_scale, KernelCenterer,
                            StandardScaler, RobustScaler, robust_scale, Normalizer, Binarizer, 
                            PolynomialFeatures, FunctionTransformer, PowerTransformer, 
                            QuantileTransformer, quantile_transform, OrdinalEncoder, OneHotEncoder, 
                            KBinsDiscretizer)
from pyxtal_ml.descriptors.descriptors import descriptor
from pyxtal_ml.datasets.collection import Collection
from pyxtal_ml.ml.ml_sklearn import method
from pyxtal_ml.ml.dl_pytorch import dl_torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from progressbar import ProgressBar
from optparse import OptionParser
import warnings
warnings.filterwarnings("ignore")

class run:
    """
    A class of production runs of pyxtal_ml.
    
    Args:
        file:
        feature:
        prop:
        N_sample:
        library:
        algo:
        feature_scaling:
        level:
        pipeline:
        hidden_layers:
    """

    def __init__(self, jsonfile, feature, prop, N_sample, library, algo, feature_scaling=False,
                 level=False, pipeline=False, hidden_layers=False):
        """
        Args:
            algo: algorithm in ['KRR', 'KNN', ....]
            feature: features among ['Chem', 'RDF', ....]
            prop: target property in ['formation_energy', 'band_gap']
            level: 'light', 'medium', 'tight'
            file: source json file
        """
        self.file = jsonfile
        self.feature0 = feature
        self.prop = prop
        self.N_sample = N_sample
        self.library = library
        self.algo = algo
        self.feature_scaling = feature_scaling
        self.level = level
        self.pipeline = pipeline
        self.hidden_layers = hidden_layers
        
        # For timing
        self.time = {}

    def load_data(self):
        """
        Obtain the struc/prop data from source.
        """
        start = time()
        self.strucs, self.props = Collection(self.file, self.prop, self.N_sample).extract_struc_prop()
        end = time()
        self.time['load_data'] = end-start

    def convert_data_1D(self, parallel=False, progress=True):
        """
        Convert the structures to descriptors in the format of 1D array.
        """
        start = time()
        if progress:
            import tqdm
            tqdm_func = tqdm.tqdm
            strucs = tqdm_func(list(self.strucs), desc=self.__class__.__name__)

        if parallel:
            from multiprocessing import Pool, cpu_count
            from functools import partial
            #QZ: it is not a good idea to use too many number of cpus due to communication
            #usually, 2-8 should be sufficient
            if type(parallel)==bool:
                ncpu = cpu_count()
            else:
                ncpu = int(parallel)
            print('---Parallel mode is on, {} cores with be used'.format(ncpu))
            with Pool(ncpu) as p:
                func = partial(self.calc_feas)
                feas = p.map(func, strucs)
                p.close()
                p.join()
        else:
            feas = []
            for struc in strucs:
                feas.append(self.calc_feas(struc))

        end = time()
        self.time['convert_data'] = end-start
        self.features = feas

    def calc_feas(self, struc, print_error=False):
        """
        Calculate user-defined features to a set of descriptors.
        
        Returns:
            an object of calculated descriptors or an empty array.
        """
        try:
            feas = descriptor(struc, self.feature0)
        except:
            feas = []
            if print_error:
                print('Problems in :', struc.formula)
        return feas

    def choose_feature(self, keys=None):
        """
        Convert the structures to descriptors in the format of 1D array.
        """
        X = []
        Y = []
        feature_scaling_length = [438, 756, 103, 1194, 541, 859, 1297]
        for fea, y0 in zip(self.features, self.props):
            if y0 != None and fea:
                X.append(fea.merge(keys=keys))
                Y.append(y0)
        
        X = np.asarray(X)
        feature_counting = fea.feature_counting
        
        if self.feature_scaling != False:
            if feature_counting in feature_scaling_length:
                X[:,:feature_counting] = self.apply_feature_scaling(X[:,:feature_counting])
            else:
                pass

        self.X = X
        self.Y = Y
        if keys is None:
            self.feature = self.feature0
        else:
            self.feature = keys
    
    def ml_train(self, algo, plot=False, print_info=True, save=False):
        """
        Build machine learning model for X/Y set.
        
        Args:
            plot: include plotting. (default = False)
            print_info: print training and results information. (default = True)
            save: save the training simulation for future usage. (default = False)
        """
        print('\nML learning with {} algorithm'.format(self.algo))
        tag = {'prop': self.prop, 'feature':self.feature}
        start = time()

        if self.library in ['SkLearn', 'sklearn']:
            ml = method(feature=self.X, prop=self.Y, algo=self.algo, tag=tag, 
                        pipeline=self.pipeline, params=self.level)
        elif self.library in ['PyTorch', 'pytorch']:
            ml = dl_torch(feature=self.X, prop=self.Y, algo=self.algo, tag=tag, 
                          hidden_layers=self.hidden_layers)
        else:
            pass
        
        end = time()
        self.time['ml'] = end-start
        if plot:
            ml.plot_correlation(figname=self.file[:-4]+'_'+self.algo+'.png')
        if print_info:
            ml.print_summary()
        if save:
            from sklearn.externals import joblib
            joblib.dump(ml.estimator, algo+'.joblib')
        self.ml = ml

    def apply_feature_scaling(self, X):
        """
        Feature scaling with the user-defined algorithm.
        Apply this function to uncorrelated arrays of feature.

        Returns:
            arrays of scaled feature.
        """
        X = eval(self.feature_scaling+'()').fit_transform(X)

        return X

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
