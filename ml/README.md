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

