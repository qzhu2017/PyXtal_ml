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
        data: a 2-D array to compute these statistics over
        axis: the axis of the array to compute the stats along

    Methods:

        get_stats:
            calculates the mean, std, kurtosis and skewness
            of a 2-D array

        mean:
            see numpy.mean

        standard_deviation:
            see numpy.std

        kurtosis:
            see scipy.stats.kurtosis

        skewness:
            see scipy.stats.skewness

        covariance:
            calculates the population covariance using numpy
            see np.cov for details
    '''

    def __init__(self, data, axis=0):
        '''
        Populate 2-D array attribute
        and axis attribute
        '''

        self._axis = axis

        if len(np.shape(data)) > 1:
            self.data = data

        else:
            # format the 1-D vector number as a 2-D row vector
            self.data = data[np.newaxis, :]


    def mean(self):
        '''
        Calculates the mean of a 2-D array along a specified axis
        '''

        return np.mean(self.data, axis=self._axis)


    def standard_deviation(self):
        '''
        Calculates the standard deviation of a 2-D array along a specified axis

        if the array length is 1, return 0 for standard deviation

        this fix is to ensure that no NaN values effect the ML models

        '''

        if np.shape(self.data) == 1:

            return 0

        else:
            return np.std(self.data, axis=self._axis)


    def kurtosis(self):
        '''
        Calculates the kurtosis of a 2-D array
        '''

        return kurtosis(self.data, axis=self._axis)

    def skewness(self):
        '''
        Calculates the skewness of a 2-D array
        '''

        return skew(self.data, axis=self._axis)

    def get_stats(self):

        '''
        Computes standardized stats over the representation array
        '''

        return np.hstack([self.mean(), self.standard_deviation(), self.kurtosis(), self.skewness()])

    def covariance(self, comparison_data):
        '''
        Computes the covariance of two feature arrays
        If the feature arrays are not of equal shape,
        the shorter feature array will be padded with zeros
        such that they are then equal length.

        Note that the covaraince matrix is symmetric, thus we only
        need the upper triangular portion of the matrix

        Args:
            comparison data: np.float, the arrays to compute the covariance matrix over
        '''

        if len(np.shape(comparison_data)) == 1:
            comparison_data = comparison_data[np.newaxis, :]

        if (np.shape(self.data) == np.array([1,1])).all() and (np.shape(comparison_data) == np.array([1,1])).all():
            print('Covariance not defined for scalars')
            raise ValueError

        elif np.shape(self.data) == np.shape(comparison_data):
            # covariance matrix
            cov_mat = np.cov(self.data, comparison_data)
            # flatten upper triangular covariance matrix
            return cov_mat[np.triu_indices(2)]

        elif np.shape(self.data)[0] > np.shape(comparison_data)[0] or np.shape(self.data)[1] > np.shape(comparison_data)[1]:

            # pad comparison vector with zeros
            new_array = np.zeros_like(self.data)
            new_array[:np.shape(comparison_data)[0], :np.shape(comparison_data)[1]] = comparison_data

            # covariance matrix
            cov_mat = np.cov(self.data, new_array)

            # flatten the upper triangular covariance matrix
            return cov_mat[np.triu_indices(2)]

        else:
            # pad self.data with zeros
            new_array = np.zeros_like(comparison_vector)
            new_array[:np.shape(self.data)[0], :np.shape(self.data)[1]] = self.data

            # covariance matrix
            cov_mat = np.cov(new_array, comparison_vector)

            # flatten the upper triangular covariance matrix
            return cov_mat[np.triu_indices(2)]
