import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use("bmh")


class method:
    """
    Collection of molecular data.
    Used for obtaining pymatgen objects from a small database file.
    """
    def __init__(self, X, Y, algo='KNN', test_size=1/3):
        self.algo = algo
        self.test_size = test_size
        self.X = X
        self.Y = Y
        self.ml()

    def ml(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_size)
        if self.algo == 'KNN':
            estimator = KNeighborsRegressor(6, weights='distance', algorithm='kd_tree', 
                                            leaf_size=40, p=1)
        elif self.algo == 'KRR':
            estimator = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=0.01, kernel='rbf', kernel_params=None)

        estimator.fit(X_train, Y_train)
        self.y_predicted = estimator.predict(X_test)
        self.y_predicted0 = estimator.predict(X_train)
        self.r2= estimator.score(X_test, Y_test, sample_weight=None)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def plot(self, figname=None, figsize=(12,8)):
        plt.figure(figsize=figsize)
        plt.scatter(self.y_predicted, self.Y_test, c='green', label='test')
        plt.scatter(self.y_predicted0, self.Y_train, c='blue', label='train')
        plt.title('{0:d} materials, r$^2$ = {1:.4f}, Algo: {2:s}'.format(len(self.Y), self.r2, self.algo))
        plt.xlabel('Prediction')
        plt.ylabel('Reference')
        plt.legend()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
