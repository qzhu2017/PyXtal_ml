# Chem+RDF+ADF+Charge (QZ)
With the following parameters in `main.py`
```python
file = 'datasets/nonmetal_MP_8049.json'
prop = 'formation_energy' #'band_gap'
feature = 'Chem+RDF+ADF+Charge'  # 'RDF', 'RDF+ADF', 'all'
algo = 'GradientBoosting'
parameters = 'light'
figname = 'test_plot.png'
N_sample = None #5000
```
We obtain the following results after commits 141
```
qiangzhu@Qiangs-MBP:~/Desktop/github/ML-Materials$ python main.py 
/Users/qiangzhu/.pyenv/versions/3.7.0/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
Time elapsed for reading the json files: 2.733 seconds
Total number of materials: 8049
The chosen feature is: Chem+RDF+ADF+Charge
0 materials have been processed
500 materials have been processed
1000 materials have been processed
1500 materials have been processed
2000 materials have been processed
2500 materials have been processed
3000 materials have been processed
3500 materials have been processed
4000 materials have been processed
4500 materials have been processed
5000 materials have been processed
5500 materials have been processed
6000 materials have been processed
6500 materials have been processed
7000 materials have been processed
7500 materials have been processed
8000 materials have been processed
Time elapsed for creating the descriptors: 784.048 seconds
Each material has 1344 descriptors
Time elapsed for machine learning: 40.209 seconds
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                  8049
Number of features:                 1344
Algorithm:              GradientBoosting
Feature:             Chem+RDF+ADF+Charge
Property:               formation_energy
r^2:                              0.9161
MAE:                              0.2217
```

![Example-01](https://github.com/qzhu2017/ML-Materials/blob/master/results/MP-8049.png)




# Chem+ADF+Charge (QZ)

It looks like the RDF does not effect the results
```
qiangzhu@Qiangs-MBP:~/Desktop/github/ML-Materials$ python main.py 
/Users/qiangzhu/.pyenv/versions/3.7.0/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
Time elapsed for reading the json files: 3.029 seconds
Total number of materials: 8049
The chosen feature is: Chem+ADF+Charge
0 materials have been processed
500 materials have been processed
1000 materials have been processed
1500 materials have been processed
2000 materials have been processed
2500 materials have been processed
3000 materials have been processed
3500 materials have been processed
4000 materials have been processed
4500 materials have been processed
5000 materials have been processed
5500 materials have been processed
6000 materials have been processed
6500 materials have been processed
7000 materials have been processed
7500 materials have been processed
8000 materials have been processed
Time elapsed for creating the descriptors: 652.704 seconds
Each material has 1284 descriptors
Time elapsed for machine learning: 41.789 seconds
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                  8049
Number of features:                 1284
Algorithm:              GradientBoosting
Feature:                 Chem+ADF+Charge
Property:               formation_energy
r^2:                              0.9144
MAE:                              0.2230
```

# Chem+Charge (QZ)
```
qiangzhu@Qiangs-MBP:~/Desktop/github/ML-Materials$ python main.py 
/Users/qiangzhu/.pyenv/versions/3.7.0/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
Time elapsed for reading the json files: 2.788 seconds
Total number of materials: 8049
The chosen feature is: Chem+Charge
0 materials have been processed
500 materials have been processed
1000 materials have been processed
1500 materials have been processed
2000 materials have been processed
2500 materials have been processed
3000 materials have been processed
3500 materials have been processed
4000 materials have been processed
4500 materials have been processed
5000 materials have been processed
5500 materials have been processed
6000 materials have been processed
6500 materials have been processed
7000 materials have been processed
7500 materials have been processed
8000 materials have been processed
Time elapsed for creating the descriptors: 246.156 seconds
Each material has 1194 descriptors
Time elapsed for machine learning: 37.384 seconds
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                  8049
Number of features:                 1194
Algorithm:              GradientBoosting
Feature:                     Chem+Charge
Property:               formation_energy
r^2:                              0.9136
MAE:                              0.2226
```

# Chem
```
qiangzhu@Qiangs-MBP:~/Desktop/github/ML-Materials$ python main.py 
/Users/qiangzhu/.pyenv/versions/3.7.0/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
Time elapsed for reading the json files: 2.761 seconds
Total number of materials: 8049
The chosen feature is: Chem
0 materials have been processed
500 materials have been processed
1000 materials have been processed
1500 materials have been processed
2000 materials have been processed
2500 materials have been processed
3000 materials have been processed
3500 materials have been processed
4000 materials have been processed
4500 materials have been processed
5000 materials have been processed
5500 materials have been processed
6000 materials have been processed
6500 materials have been processed
7000 materials have been processed
7500 materials have been processed
8000 materials have been processed
Time elapsed for creating the descriptors: 93.788 seconds
Each material has 438 descriptors
Time elapsed for machine learning: 17.714 seconds
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                  8049
Number of features:                  438
Algorithm:              GradientBoosting
Feature:                            Chem
Property:               formation_energy
r^2:                              0.9153
MAE:                              0.2197
```

# Charge
```
qiangzhu@Qiangs-MBP:~/Desktop/github/ML-Materials$ python main.py 
/Users/qiangzhu/.pyenv/versions/3.7.0/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
Time elapsed for reading the json files: 2.764 seconds
Total number of materials: 8049
The chosen feature is: Charge
0 materials have been processed
500 materials have been processed
1000 materials have been processed
1500 materials have been processed
2000 materials have been processed
2500 materials have been processed
3000 materials have been processed
3500 materials have been processed
4000 materials have been processed
4500 materials have been processed
5000 materials have been processed
5500 materials have been processed
6000 materials have been processed
6500 materials have been processed
7000 materials have been processed
7500 materials have been processed
8000 materials have been processed
Time elapsed for creating the descriptors: 134.697 seconds
Each material has 756 descriptors
Time elapsed for machine learning: 15.699 seconds
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                  8049
Number of features:                  756
Algorithm:              GradientBoosting
Feature:                          Charge
Property:               formation_energy
r^2:                              0.5916
MAE:                              0.5215
```
