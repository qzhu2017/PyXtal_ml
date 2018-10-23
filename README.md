# PyXtal\_ml

## Overview
A library of ML training of materials' properties
- `datasets`: python class to download the data from open database + data in json format
- `descriptors`: python class for different types of descriptors (RDF, ADF, chemical labeling and enviroments)
- `ml`: python class for the choice of different pipelines of ML methods (KRR, ...)
- `test`: python class for unit test (to implement)
- `results`: a collection of results.

## Hierarchy
This code has the following hierarchy
```
pyxtal_ml
├── main.py
├── descriptors -> descriptors.py -> (RDF.py, ADF.py, DDF.py, Chem.py, .etc)
├── datasets -> collection.py -> (json files)
├── ml -> methods.py (KRR, KNN, ANN, .etc.)
├── test -> (various scripts to test the accuracy and efficiency of the code)
```

## Installation
```
# git clone https://github.com/qzhu2017/PyXtal_ml.git
# python setup.py install
```

## Dependencies:
* [SciPy 1.0.1](https://www.scipy.org/install.html)
* [NumPy 1.14.3](https://www.scipy.org/scipylib/download.html)
* [Scikit-learn](http://scikit-learn.org/stable/)
* [Pandas 0.20.3](https://pandas.pydata.org/getpandas.html)
* [Pymatgen](http://pymatgen.org/#getting-pymatgen)
