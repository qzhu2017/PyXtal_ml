from pyxtal_ml.run import run
from pkg_resources import resource_filename

# Please define your values in here, option 1, and option 2.
jsonfile = resource_filename("pyxtal_ml", "datasets/NaGaAu.json")
feature = 'bispectrum+element+covariance'
feature_scaling = False #'MinMaxScaler'
prop = 'formation_energy'
N_sample = 100
library = 'SkLearn' # SkLearn
algorithm = 'GB' # or dl

# Option 1: If you want to use an algorithm from Scikit-learn, 
# please enter the following
level = 'light'
pipeline = False

# Option 2: If you want to use an algorithm from PyTorch, 
# please enter the following
hidden_layers = {"n_layers": 5, "n_neurons": [100]}
conv_layers = {"conv": ['Conv2d(1,3,2)', 'Conv2d(3,5,5)'],
                "maxpool": ['MaxPool2d(4)']}

# Running the user-defined values. Don't tresspass beyond this point.
runner = run(jsonfile=jsonfile, 
             feature=feature, 
             prop=prop, 
             N_sample=N_sample, 
             feature_scaling=feature_scaling)
runner.load_data()
runner.convert_data_1D(parallel=8) #choose cpu number
runner.choose_feature(keys=feature) #choose feature combinations if you want
runner.ml_train(library=library, 
                 algo=algorithm, 
                 level=level, 
                 pipeline=pipeline, 
                 hidden_layers=hidden_layers, 
                 conv_layers=conv_layers)
runner.print_time()
