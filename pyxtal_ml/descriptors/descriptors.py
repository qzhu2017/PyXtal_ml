import numpy as np
from pymatgen.core.structure import Structure
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
from pyxtal_ml.descriptors.DDF import DDF
from pyxtal_ml.descriptors.prdf import PRDF
from pyxtal_ml.descriptors.voronoi_descriptors import Voronoi_Descriptors
from pyxtal_ml.descriptors.crystal_graph import crystalgraph
from pyxtal_ml.descriptors.bond_order_params import steinhardt_params
from pyxtal_ml.descriptors.power_spectrum import power_spectrum


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
                   'RDF', 'ADF', 'DDF', 'PRDF', 'bond_order', 'power_spectrum']
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
                elif lib == 'DDF':
                    self.descriptor['DDF'] = DDF(self.struc).DDF
                    self.descriptor['DDF'] = self.apply_feature_scaling_array(
                        self.descriptor['DDF'])
                elif lib == 'Chem':
                    self.descriptor['Chem'] = Chem(self.struc).mean_chem
                elif lib == 'Charge':
                    self.descriptor['Charge'] = Charge(self.struc).mean_chg
                elif lib == 'Voronoi':
                    self.descriptor['Voronoi'] = Voronoi_Descriptors(
                        self.struc).all()
                elif lib == 'PRDF':
                    self.descriptor['PRDF'] = PRDF(self.struc).PRDF
                    self.descriptor['PRDF'] = self.apply_feature_scaling_array(
                        self.descriptor['PRDF'])
                elif lib == 'cg':
                    self.descriptor['cg'] = crystalgraph(
                        self.struc).crystal_graph
                elif lib == 'bond_order':
                    self.descriptor['bond_order'] = steinhardt_params(
                        self.struc).params
                elif lib == 'power_spectrum':
                    self.descriptor['power_spectrum'] = power_spectrum(
                        self.struc).Power_spectrum

        else:
            for lib in self.libs:
                if lib == 'RDF':
                    self.descriptor['RDF'] = RDF(self.struc).RDF[1, :]
                elif lib == 'ADF':
                    self.descriptor['ADF'] = ADF(self.struc).all
                elif lib == 'DDF':
                    self.descriptor['DDF'] = DDF(self.struc).DDF
                elif lib == 'Chem':
                    self.descriptor['Chem'] = Chem(self.struc).mean_chem
                elif lib == 'Charge':
                    self.descriptor['Charge'] = Charge(self.struc).mean_chg
                elif lib == 'Voronoi':
                    self.descriptor['Voronoi'] = Voronoi_Descriptors(
                        self.struc).all()
                elif lib == 'PRDF':
                    self.descriptor['PRDF'] = PRDF(self.struc).PRDF
                elif lib == 'cg':
                    self.descriptor['cg'] = crystalgraph(
                        self.struc).crystal_graph
                elif lib == 'bond_order':
                    self.descriptor['bond_order'] = steinhardt_params(
                        self.struc).params
                elif lib == 'power_spectrum':
                    self.descriptor['power_spectrum'] = power_spectrum(
                        self.struc).Power_spectrum

    def merge(self, keys=None):
        if keys is None:
            keys0 = self.descriptor.keys()
        else:
            keys0 = []
            for key in keys.split('+'):
                keys0.append(key)

        keys0, self.feature_counting = self.sort_features(keys0)

        arr = []
        for key in keys0:
            if len(self.descriptor[key]) == 0:
                print(key, np.shape(self.descriptor[key]))
            if len(arr) == 0:
                arr = self.descriptor[key]
            else:
                arr = np.hstack((arr, self.descriptor[key]))
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
    des = descriptor(test)
    for lib in des.libs:
        print(lib, len(des.descriptor[lib]))
    print(des.merge())
    print('length of the descriptors: ', np.shape(des.merge()))
