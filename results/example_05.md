# PRDF as the sole descriptor

## Training on 1000 materials
```
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                  1000
Number of features:                 4560
Algorithm:                            RF
Feature:                            PRDF
Property:               formation_energy
r^2:                              0.7621
MAE:                              0.3268
Parameters: {'n_estimators': 10}
Mean train_score:                 0.9321
Std train_score:                  0.0154
load_data                   1.26 seconds
convert_data              136.66 seconds
ml                          3.45 seconds
```

## Training on 8049 materials
```
ML learning with RF algorithm
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                  8049
Number of features:                 4560
Algorithm:                            RF
Feature:                            PRDF
Property:               formation_energy
r^2:                              0.8958
MAE:                              0.2312
Parameters: {'n_estimators': 10}
Mean train_score:                 0.9775
Std train_score:                  0.0016
load_data                   2.93 seconds
convert_data              142.96 seconds
ml                         22.63 seconds
```

# PRDF+Chem

## Training on 8049 materials

```
ML learning with RF algorithm
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                  8049
Number of features:                 4998
Algorithm:                            RF
Feature:                       PRDF+Chem
Property:               formation_energy
r^2:                              0.9303
MAE:                              0.1747
Parameters: {'n_estimators': 10}
Mean train_score:                 0.9868
Std train_score:                  0.0008
load_data                   2.90 seconds
convert_data              146.13 seconds
ml                         56.95 seconds
```

## Training on 25923 materials from Jarvis
```
ML learning with RF algorithm
----------------------------------------
-------------SUMMARY of ML--------------
----------------------------------------
Number of samples:                 25923
Number of features:                 4998
Algorithm:                            RF
Feature:                       PRDF+Chem
Property:               formation_energy
r^2:                              0.9316
MAE:                              0.1589
Parameters: {'n_estimators': 10}
Mean train_score:                 0.9843
Std train_score:                  0.0010
load_data                   8.41 seconds
convert_data              533.22 seconds
ml                        254.84 seconds
```
