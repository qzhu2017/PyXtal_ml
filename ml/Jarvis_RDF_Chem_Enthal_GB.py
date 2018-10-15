# Gradient Boosting Regression with Random Forest algorithm
# is employed to predict Formation Energy of materials with
# RDF as the descriptor. Dataset is taken from Jarvis.

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from monty.json import MontyEncoder, MontyDecoder
from monty.serialization import loadfn, dumpfn
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import numpy as np
import sys
sys.path.append('../')
from descriptors.RDF import *
from descriptors.chem import *

# Functions
def isfloat(value):
    #Simple function to check if the data is available/float
    try:
        float(value)
        return True
    except ValueError:
        return False

def get_features(data):
    X1 = []  #RDF
    X2 = []  #Chem
    X = [] #RDF + Chem
    Y = []  #Formation energy
    for i in data:
        y = i['form_enp']
        if isfloat(y):
            crystal = i['final_str']
            X1.append(RDF(crystal).RDF[1,:])
            X2.append(Chem(crystal).mean_chem)
            Y.append(y)
    X1 = np.array(X1).astype(np.float)
    X2 = np.array(X2).astype(np.float)
    X = [X1,X2]
    return X, Y

# Import data
data = loadfn('../datasets/jdft_3d-7-7-2018.json',cls=MontyDecoder)

# Split to train and test sets
X, Y = get_features(data[:5])
#X=np.array(X).astype(np.float)
Y=np.array(Y).astype(np.float)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)

# Perform gradient boosting
est= GradientBoostingRegressor(loss = 'huber')
varthres = VarianceThreshold(0.01)
pipe=Pipeline([("fs", varthres),("est", est)])
param_grid = {
    'est__learning_rate': [0.1],
    'est__n_estimators': [1000]}
search = GridSearchCV(pipe, param_grid, cv=10,iid=False, return_train_score=False)
search.fit(X_train,Y_train)
print(search.best_params_)

best_learning = search.best_params_['est__learning_rate']
best_estimators = search.best_params_['est__n_estimators']

est2=GradientBoostingRegressor(loss='huber', learning_rate = best_learning, n_estimators = best_estimators)
pipe2=Pipeline([("fs", VarianceThreshold(threshold = 0.01)),("est", est2)])
pipe2.fit(X_train,Y_train)
Y_pred = pipe2.predict(X_test)
r2=pipe2.score(X_test,Y_test,sample_weight=None)
print('r^2= ', r2)

mae = mean_absolute_error(Y_pred,Y_test)
print('mae = ',mae)

# Plotting
plt.plot(Y_test, Y_pred, 'bo')
plt.xlabel('Enthalpy_dft (eV/atom)')
plt.ylabel('Enthalpy_ML (eV/atom)')
plt.savefig('Results/enthalpy_form.png')
