import numpy as np
from pymatgen.core.structure import Structure
from itertools import combinations
from optparse import OptionParser
from sklearn.preprocessing import (MinMaxScaler, minmax_scale, MaxAbsScaler, maxabs_scale, KernelCenterer,
                                   StandardScaler, RobustScaler, robust_scale, Normalizer, Binarizer,
                                   PolynomialFeatures, FunctionTransformer, PowerTransformer,
                                   QuantileTransformer, quantile_transform, OrdinalEncoder, OneHotEncoder,
                                   KBinsDiscretizer)
from pyxtal_ml.descriptors.RDF import RDF
from pyxtal_ml.descriptors.ADF import ADF
from pyxtal_ml.descriptors.chem import Chem
from pyxtal_ml.descriptors.charge import Charge
from pyxtal_ml.descriptors.prdf import PRDF
from pyxtal_ml.descriptors.voronoi_descriptors import Voronoi_Descriptors
from pyxtal_ml.descriptors.bond_order_params import steinhardt_params
from pyxtal_ml.descriptors.power_spectrum import power_spectrum
from pyxtal_ml.descriptors.bispectrum import Bispectrum
from pyxtal_ml.descriptors.stats import descriptor_stats
from pyxtal_ml.descriptors.Element import element_attributes


class descriptor:
    """Collection of molecular data.
    Used for obtaining pymatgen objects from a small database file.

    Example of use:
    >>> from ml_materials.descriptors import descriptor
    >>> des=descriptor(struc, 'all').merge()
    Args:
        name: the type of collection to get. Defaults to "molecules"
    """

    def __init__(self, crystal, libs='all', feature_scaling=False):
        self.libs = libs
        self.struc = crystal
        self.feature_scaling = feature_scaling
        self.descriptor = {}
        options = ['Chem', 'Voronoi', 'Charge',
                   'RDF', 'ADF', 'DDF', 'PRDF', 'bond_order', 'power_spectrum', 'bispectrum', 'element']
        self.libs = []
        if libs == 'all':
            self.libs = options
        else:
            for lib in libs.split('+'):
                self.libs.append(lib)

        if self.feature_scaling != False:
            for lib in self.libs:
                if lib == 'RDF':
                    self.descriptor['RDF'] = RDF(self.struc).RDF[1, :]
                    self.descriptor['RDF'] = self.apply_feature_scaling_array(
                        self.descriptor['RDF'])
                elif lib == 'ADF':
                    self.descriptor['ADF'] = ADF(self.struc).all
                    self.descriptor['ADF'] = self.apply_feature_scaling_array(
                        self.descriptor['ADF'])
                elif lib == 'Chem':
                    self.descriptor['Chem'] = Chem(self.struc).chem_stats
                elif lib == 'Charge':
                    self.descriptor['Charge'] = Charge(self.struc).chg_stats
                elif lib == 'Voronoi':
                    self.descriptor['Voronoi'] = Voronoi_Descriptors(
                        self.struc).all()
                elif lib == 'PRDF':
                    self.descriptor['PRDF'] = PRDF(self.struc).PRDF
                    self.descriptor['PRDF'] = self.apply_feature_scaling_array(
                        self.descriptor['PRDF'])
                elif lib == 'bond_order':
                    self.descriptor['bond_order'] = steinhardt_params(
                        self.struc).params
                elif lib == 'power_spectrum':
                    self.descriptor['power_spectrum'] = power_spectrum(
                        self.struc).Power_spectrum
                elif lib == 'bispectrum':
                    f = Bispectrum(self.struc)
                    self.descriptor['bispectrum'] = f.get_descr()
                elif lib == 'element':
                    self.descriptor['element'] = element_attributes(
                        self.struc).properties

        else:
            for lib in self.libs:
                if lib == 'RDF':
                    self.descriptor['RDF'] = RDF(self.struc).RDF[1, :]
                elif lib == 'ADF':
                    self.descriptor['ADF'] = ADF(self.struc).all
                elif lib == 'Chem':
                    self.descriptor['Chem'] = Chem(self.struc).chem_stats
                elif lib == 'Charge':
                    self.descriptor['Charge'] = Charge(self.struc).chg_stats
                elif lib == 'Voronoi':
                    self.descriptor['Voronoi'] = Voronoi_Descriptors(
                        self.struc).all()
                elif lib == 'PRDF':
                    self.descriptor['PRDF'] = PRDF(self.struc).PRDF
                elif lib == 'bond_order':
                    self.descriptor['bond_order'] = steinhardt_params(
                        self.struc).params
                elif lib == 'power_spectrum':
                    self.descriptor['power_spectrum'] = power_spectrum(
                        self.struc).Power_spectrum
                elif lib == 'bispectrum':
                    f = Bispectrum(self.struc)
                    self.descriptor['bispectrum'] = f.get_descr()
                elif lib == 'element':
                    self.descriptor['element'] = element_attributes(
                        self.struc).properties

        if 'covariance' in self.libs:
            # descriptor dictionary keys
            features = self.descriptor.keys()
            cov = []
            # all feature pairwise combinations
            for feat in combinations(features, 2):
                # populate feature arrays
                feature_1 = self.descriptor[feat[0]]
                feature_2 = self.descriptor[feat[1]]

                '''
                Covariance only supports 2-D data,
                if the feature is 1-D add an axis and calculate covariance
                if the feature is more than 2-D, continue to next iteration
                if the feature is 2-D calculate covariance
                '''
                if len(np.shape(feature_1)) == 1:
                    feature_1 = feature_1[np.newaxis, :]

                elif len(np.shape(feature_1)) > 2:
                    print('Covariance only supports 2-D data')
                    raise ValueError

                if len(np.shape(feature_2)) == 1:
                    feature_2 = feature_2[np.newaxis, :]

                elif len(np.shape(feature_2)) > 2:
                    print('Covariance only supports 2-D data')
                    raise ValueError

                '''
                Compute the covariance between row wise combinations
                iterate over rows
                '''
                for index_1 in np.arange(0, np.shape(feature_1)[1]):
                    row_1 = feature_1[:, index_1]
                    for index_2 in np.arange(0, np.shape(feature_2)[1]):
                        row_2 = feature_2[: ,index_2]

                        cov.append(descriptor_stats(row_1).covariance(row_2))

            self.descriptor['covariance'] = np.array(cov)


    def merge(self, keys=None):
        if keys is None:
            keys0 = self.descriptor.keys()
        else:
            keys0 = []
            for key in keys.split('+'):
                keys0.append(key)

        keys0, self.feature_counting = self.sort_features(keys0)

        non_stats_keys = ['RDF', 'ADF', 'PRDF', 'Voronoi', 'covariance']
        arr = []
        for key in keys0:
            if len(self.descriptor[key]) == 0:
                print(key, np.shape(self.descriptor[key]))
            if len(arr) == 0:

                if key in non_stats_keys:
                    arr = self.descriptor[key].flatten()

                else:
                    arr = descriptor_stats(self.descriptor[key]).get_stats().flatten()
            else:

                if key in non_stats_keys:
                    arr = np.hstack((arr, self.descriptor[key].flatten()))

                else:
                    arr = np.hstack((arr, descriptor_stats(self.descriptor[key]).get_stats().flatten()))
        return arr

    def sort_features(self, features):
        options = ['Chem', 'Voronoi', 'Charge']
        feature_counting = 0
        featuring_options = []
        featuring_noptions = []
        for feat in features:
            if feat == 'Chem':
                featuring_options.append(feat)
                feature_counting += 438
            elif feat == 'Charge':
                featuring_options.append(feat)
                feature_counting += 756
            elif feat == 'Voronoi':
                featuring_options.append(feat)
                feature_counting += 103
            else:
                featuring_noptions.append(feat)
        feature = featuring_options+featuring_noptions

        return features, feature_counting

    def apply_feature_scaling_array(self, X):
        """
        Feature scaling with the user-defined algorithm.
        Apply this function to correlated arrays of feature.
        E.g. partial radial distribution function.

        Returns:
            arrays of scaled feature.
        """
        X = eval(self.feature_scaling+'()').fit_transform(X)

        return X


if __name__ == "__main__":
    # -------------------------------- Options -------------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest="structure", default='',
                      help="crystal from file, cif or poscar, REQUIRED",
                      metavar="crystal")
    parser.add_option("-d", "--descriptors", dest="descriptors", default='all',
                      help="descriptors, all, rdf+adf")

    (options, args) = parser.parse_args()
    if options.structure.find('cif') > 0:
        fileformat = 'cif'
    else:
        fileformat = 'poscar'

    test = Structure.from_file(options.structure)
    des = descriptor(test, 'bispectrum')
    for lib in des.libs:
        print(lib, len(des.descriptor[lib]))
    print(des.merge())
    print('length of the descriptors: ', np.shape(des.merge()))
