import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('X_ICSD.txt', dtype = 'float')
Y = np.loadtxt('Y_ICSD.txt', dtype = 'float')

Rank = np.argsort(Y)
Y = Y[Rank[0:5000]]
X = X[Rank[0:5000], :40]
#from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3)

#estimator = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=0.01, kernel='rbf', kernel_params=None)
estimator = KNeighborsRegressor(6, weights='distance', algorithm='kd_tree', leaf_size=40, p=1)

estimator.fit(X_train, Y_train)
y_predicted = estimator.predict(X_test)

r2= estimator.score(X_test, Y_test, sample_weight=None)
print('r^2 = ', r2)

plt.scatter(y_predicted, Y_test, c='green')
plt.title('DOS: Actual vs Predicted-- 9320 Materials')
plt.xlabel('y_predicted')
plt.ylabel('Y_test')
plt.savefig('KNN.png')
plt.show()
#np.savetxt('/scratch/yanxonh/python', r2)
