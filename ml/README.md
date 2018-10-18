We provide the interface to many ML algorithms.
- Nearest Neighbor: `KNN`
- Kernel Ridge Regression: `KRR`
- Random Forest: `RF`
- SVM: `SVM`
- Gradient Boosting: `GradientBoosting`
- LASSO: `LASSO`
- Stochastic gradient descent: `StochasticGD`
- Artificial Neural Network: `ANN`

Typically, one needs to explore the parameter space for each algorithm in order to achieve the best performance. 
Therefore, we provide three sets of trainings to different algorithm.
- `light`: typically just use the default parameters provided by the ML algorithm.
- `medium`: a medium grid search for the optimal parameters
- `tight`: a extensive grid search for the optimal parameters

The default parameter sets are defined in [`default_params.yml`](https://github.com/qzhu2017/ML-Materials/blob/master/ml/default_params.yml). It is **human redable and edible**. One can simply modify this set of parameter if needed. On the other hand, one can pass the parameters as the dictionary when calling `ml.method`, these parameters will **override** the parameters used in the existing set if there exists a match. 

