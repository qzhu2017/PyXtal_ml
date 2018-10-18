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
- `light`: typically just use the `default parameters` provided by each ML algorithm. This allows a minimum efforts to complete the model training process. This is useful for quick and dirty test. If the dataset is good, one may still get good fitting from such a brute-force training.
- `medium`: `default parameters` + `cross_validation`. To run training multiple times and obtain the averaged score of fitting performance. This is recommended for real application.
- `tight`: `grid search` + `cross_validation`. This allows an exhaustive exploration to achieve the best performance. But it will require a significant of computation time.

The default parameter sets are defined in [`default_params.yml`](https://github.com/qzhu2017/ML-Materials/blob/master/ml/default_params.yml). It is **human redable and edible**. One can simply modify this set of parameter if needed. On the other hand, one can pass the parameters as the dictionary when calling `ml.method`, these parameters will **override** the parameters used in the existing set if there exists a match. 

