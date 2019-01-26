import numpy as np
from scipy.stats import kurtosis, skew

class descriptor_stats(object):
    '''
    A class containing standardized statistics to compute over each
    representation

    These statistics include:
        mean, standard deviation, kurtosis, and skewness

    Population covariance is also considered separately

    Args:
        vect: a 1-D array to compute these statistics over

    Methods:

        get_stats:
            calculates the mean, std, kurtosis and skewness
            of a vector

        mean:
            see numpy.mean

        standard_deviation:
            see numpy.std

        kurtosis:
            see scipy.stats.kurtosis

        skewness:
            see scipy.stats.kurtosis

        covariance:
            calculates the population covariance using numpy
            see np.cov for details
    '''

    def __init__(self, vect):
        '''
        Populate vector attribute
        '''

        self.vector = vect


    def mean(self):
        '''
        Calculates the mean of the vector
        '''

        return np.mean(self.vector)


    def standard_deviation(self):
        '''
        Calculates the standard deviation of a vector

        if the vector length is 1, return 0 for standard deviation

        this fix is to ensure that no NaN values effect the ML models

        '''

        if len(self.vector) == 1:

            return 0

        else:
            return np.std(self.vector)


    def kurtosis(self):
        '''
        Calculates the kurtosis of a vector
        '''

        return kurtosis(self.vector)

    def skewness(self):
        '''
        Calculates the skewness of a vector
        '''

        return skew(self.vector)

    def get_stats(self):

        '''
        Computes standardized stats over the representation vector
        '''

        return [self.mean(), self.standard_deviation(), self.kurtosis(), self.skewness()]

    def covariance(self, comparison_vector):
        '''
        Computes the covariance of two feature vectors
        If the feature vectors are not of equal length,
        the shorter feature vector will be padded with zeros
        such that they are then equal length.

        Note that the covaraince matrix is symmetric, thus we only
        need the upper triangular portion of the matrix

        Args:
            comparison vector: np.float, the vector to compute the covariance matrix over
        '''

        if len(self.vector) == 1 and len(comparison_vector) == 1:
            print('Covariance not defined for scalars')
            raise ValueError

        elif len(self.vector) == len(comparison_vector):
            # covariance matrix
            cov_mat = np.cov(self.vector, comparison_vector)
            # flatten upper triangular covariance matrix
            return cov_mat[np.triu_indices(2)]

        elif len(self.vector) > len(comparison_vector):

            # pad comparison vector with zeros
            new_vect = np.zeros_like(self.vector)
            new_vect[:len(comparison_vector)] = comparison_vector

            # covariance matrix
            cov_mat = np.cov(self.vector, new_vect)

            # flatten the upper triangular covariance matrix
            return cov_mat[np.triu_indices(2)]

        else:
            # pad self.vector with zeros
            new_vect = np.zeros_like(comparison_vector)
            new_vect[:len(self.vector)] = self.vector

            # covariance matrix
            cov_mat = np.cov(new_vect, comparison_vector)

            # flatten the upper triangular covariance matrix
            return cov_mat[np.triu_indices(2)]
