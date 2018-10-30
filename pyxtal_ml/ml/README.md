We provide the interface to many ML algorithms.

Currently, we focus on the models in `sklearn`
- Nearest Neighbor: `KNN`
- Kernel Ridge Regression: `KRR`
- Random Forest: `RF`
- Support Vector Regression: `SVR`
- Gradient Boosting: `GradientBoosting`
- LASSO: `LASSO`
- ElasticNet: `ENet`
- Stochastic gradient descent: `StochasticGD`
- Artificial Neural Network: `ANN`

In addition, we offer add-on models to complement the models above:
- VarianceThreshold: `VT`
- Principal components analysis: `PCA`

These add-on models aren't a necessary steps to achieve accurate machine learning predictions. However, these models can increase the accuracy of predictions by manipulating the training input data (i.e. training descriptors). The manipulation involves getting rid of non-contributing descriptors before the training of machine learning algorithm happened. To emphasize, these add-on models only apply to the training descriptors, and this is accomplished through pipeline method from `sklearn`.

Pipeline method applies an add-on model and a machine learning algorithms in order. For example, VarianceThreshold and Random Forest are picked to predict a materials' property. First, pipeline method will apply VarianceThreshold to the training descriptors, i.e. VarianceThreshold removes non-contributing descriptors. Non-contributing descriptors are the descriptors that have variance equals zero by default. Second, pipeline method will perform Random Forest algorithm with the contributing descriptors. These add-on models have the advantage of reducing the size of descriptors without sacrificing any information. Thus, the training time is reduced based on dimensionality factor.

We are also going to explore the Deep Learning methods from `Keras` and `Pytorch` soon.

Typically, one needs to explore the parameter space for each algorithm in order to achieve the best performance. 
Therefore, we provide a set of three training levels: light, medium, and tight.
A standard ML fitting involves these consecutive steps:
- split the data to training and test sets
- choose the estimator (`algo`)
- explore the parameter space
- process the features 
- fit and predict
- cross validation

However, the accuracy/efficiency of ML training is really proceedure and parameter dependent. Therefore we generally provide the following three set of training stratergies. 

- **`light`**: `default parameters` provided by each ML algorithm with single run of training. This allows a minimum effort to complete the model training. It is useful for quick and dirty test. If the dataset is good, one may still get good fitting from such a brute-force training.

- **`medium`**: `default parameters` + `cross_validation`. To run training multiple times and obtain the averaged score of fitting performance. This is recommended for real application.

- **`tight`**: `grid search` + `cross_validation`. This allows an exhaustive exploration to achieve the best performance. But it will require a significant amount of computation time. One has to be very patient!

The default parameter sets are defined in [`default_params.yaml`](https://github.com/qzhu2017/PyXtal_ml/blob/master/pyxtal_ml/ml/default_params.yaml). It is **human readable and editable**. One can simply modify this set of parameter if needed. On the other hand, one can pass the parameters as the dictionary when calling `ml.method`, these parameters will **override** the parameters used in the existing set if there exists a match. 
