We offer the interface to many machine learning regression algorithms from [Scikit-learn](http://scikit-learn.org/stable/) to predict materials' properties.

Currently, the available models are:
- K-Nearest Neighbor: `KNeighborsRegressor` or `KNN`
- Kernel Ridge: `KernelRidge` or `KRR`
- Support Vector Machine: `SVR`
- GradientBoosting: `GradientBoostingRegressor` or `GB`
- RandomForest: `RandomForestRegressor` or `RF`
- LASSO: `Lasso`
- ElasticNet: `ElasticNet` or `ENet`
- Stochastic Gradient Descent: `SGDRegressor` or `SGD`
- Artificial Neural Network: `MLPRegressor` or `ANN`
- Gaussian Process Regressor: `GaussianProcessRegressor` or `GPR`

In addition, we provide add-on models to complement the models above:
- VarianceThreshold: `VarianceThreshold` or `VT`
- Principal components analysis: `PCA`

**Note: The naming is adopted from Scikit-learn and our short notations. In artificial neural network, `MLPRegressor` is the term defined by Scikit-learn for ANN regressor, and `ANN` is our short-hand notation.

These add-on models aren't a necessary steps to complete accurate machine learning predictions. However, these models can increase the accuracy of predictions by manipulating the training input data (i.e. training descriptors). The manipulation involves getting rid of non-contributing descriptors before the training of machine learning algorithm happened. To emphasize, these add-on models only apply to the training descriptors, and this is accomplished through pipeline method from `sklearn`.

Pipeline method applies an add-on model and a machine learning algorithms in order. For example, VarianceThreshold and Random Forest are picked to predict a materials' property. First, pipeline method will apply VarianceThreshold to the training descriptors, i.e. VarianceThreshold removes non-contributing descriptors. Non-contributing descriptors are the descriptors that have variance equals zero by default. Second, pipeline method will perform Random Forest algorithm with the contributing descriptors. These add-on models have the advantage of reducing the size of descriptors without sacrificing any information. Thus, the training time is reduced based on dimensionality factor.

A standard ML fitting involves these consecutive steps:
- split the data into training and test sets
- choose an estimator (`algo`)
- process the features
- explore the parameter space 
- cross validation
- fit and predict

However, the accuracy/efficiency of ML training is really procedure- and parameter-dependent. Therefore, we generally provide the following a set of three training levels and user-defined parameters: 

- **`light`**: `default parameters` provided by [Scikit-learn](http://scikit-learn.org/stable/) for each ML algorithm with single run of training. This allows a minimum effort to complete the model training. It is useful for quick and dirty test. If the dataset is good, users may still get good fitting from such a brute-force training.

- **`medium`**: `default parameters` + `cross_validation`. Cross-validate training with k-fold cross-validation. For `medium`, k-fold equals 10.

- **`tight`**: `grid search` + `cross_validation`. This allows an exhaustive exploration to achieve the best performance. But it will require a significant amount of computation time. Users has to be very patient!

- **`user-defined`**: `users can defined their own k-fold cross-validation >= 2 and parameters`. For example, users can pass a dictionary in this format, {'cv':5, 'my_parameters': {"learning_rate": [0.1,0.2]}}. Although the names 'cv' and 'my_parameters' do not matter, they need to be in that order, i.e. k-fold cross-validation and parameters.

Another user-defined-parameters path is by changing the value in the [`default_params.yaml`](https://github.com/qzhu2017/PyXtal_ml/blob/master/pyxtal_ml/ml/default_params.yaml). The default_params.yaml is **human readable and editable**. Users can simply modify this set of parameter if needed. If users pass the parameters as dictionary, the default_params.yaml will be **overriden**.

### COMING SOON: Deep Learning methods with `Keras` and `Pytorch`.
