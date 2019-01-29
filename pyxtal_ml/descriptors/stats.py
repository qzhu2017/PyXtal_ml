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

        '''
        The data array should be at least 2 dimensional
        if it is 1-dimensional, simply add an axis.

        If the data is a scalar or 0-dimensional, in our case
        this corresponds to a structure with a single periodic site
        then we must copy the data in another manner
        '''
        if type(data) != np.ndarray:
            data = np.array(data)

        if len(np.shape(data)) > 1:
            self.data = data

        else:
            if np.shape(data) == ():
                data = np.array([data, data])

            self.data = data[:, np.newaxis]


    def mean(self):
        '''
        Calculates the mean of a 2-D array along a specified axis
        '''

        return np.mean(self.data, axis=self._axis)

    def min(self):
        '''
        Calculates the minimum value of an array along a specied axis
        '''
        return np.amin(self.data, axis=self._axis)

    def max(self):
        '''
        Calculates the maximum value of an array along a specied axis
        '''
        return np.amax(self.data, axis=self._axis)

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

        stats = np.hstack([[self.mean()], [self.min()], [self.max()], [self.standard_deviation()], [self.kurtosis()], [self.skewness()]])

        if self._axis == 0:
            return np.reshape(stats, (6, np.shape(self.data)[1])).T

        elif self._axis == 1:
            return np.reshape(stats, (6, np.shape(self.data)[0])).T


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

        if type(comparison_data) != np.ndarray:
            comparison_data = np.array(comparison_data)

        if len(np.shape(comparison_data)) > 1:
            comparison_data = comparison_data

        else:
            if np.shape(comparison_data) == ():
                comparison_data = np.array([comparison_data, comparison_data])

            comparison_data = comparison_data[:, np.newaxis]

        if (np.shape(self.data) == np.array([1,1])).all() and (np.shape(comparison_data) == np.array([1,1])).all():
            print('Covariance not defined for scalars')
            raise ValueError

        elif np.shape(self.data) == np.shape(comparison_data):
            # covariance matrix
            cov_mat = np.cov(self.data, comparison_data, rowvar=False)
            # flatten upper triangular covariance matrix
            return cov_mat[0,1]

        elif np.shape(self.data)[0] >= np.shape(comparison_data)[0] and np.shape(self.data)[1] >= np.shape(comparison_data)[1]:

            # pad comparison vector with zeros
            new_array = np.zeros_like(self.data)
            new_array[:np.shape(comparison_data)[0], :np.shape(comparison_data)[1]] = comparison_data

            # covariance matrix
            cov_mat = np.cov(self.data, new_array, rowvar=False)

            # flatten the upper triangular covariance matrix
            return cov_mat[0,1]

        elif np.shape(self.data)[0] <= np.shape(comparison_data)[0] and np.shape(self.data)[1] >= np.shape(comparison_data)[1]:
            # pad self.data with necessary zeros
            new_data_array = np.zeros([np.shape(comparison_data)[0], np.shape(self.data)[1]])
            new_data_array[:np.shape(self.data)[0], :np.shape(self.data)[1]] = self.data

            # pad comparison data with necessary zeroes

            new_comparison_array = np.zeros([np.shape(comparison_data)[0], np.shape(self.data)[1]])
            new_comparison_array[:np.shape(comparison_data)[0], :np.shape(comparison_data)[1]] = comparison_data

            cov_mat = np.cov(new_data_array, new_comparison_array, rowvar=False)

            return cov_mat[0,1]

        elif np.shape(self.data)[0] >= np.shape(comparison_data)[0] and np.shape(self.data)[1] <= np.shape(comparison_data)[1]:
            # pad with necessary zeros
            new_data_array = np.zeros([np.shape(self.data)[0], np.shape(comparison_data)[1]])
            new_data_array[:np.shape(self.data)[0], :np.shape(self.data)[1]] = self.data

            new_comparison_array = np.zeros([np.shape(self.data)[0], np.shape(comparison_data)[1]])
            new_comparison_array[:np.shape(comparison_data)[0], :np.shape(comparison_data)[1]] = comparison_data

            cov_mat = np.cov(new_data_array, new_comparison_array, rowvar=False)

            return cov_mat[0,1]

        else:
            # pad self.data with zeros
            new_array = np.zeros_like(comparison_data)
            new_array[:np.shape(self.data)[0], :np.shape(self.data)[1]] = self.data

            # covariance matrix
            cov_mat = np.cov(new_array, comparison_data, rowvar=False)

            # flatten the upper triangular covariance matrix
            return cov_mat[0,1]
