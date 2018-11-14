from pyxtal_ml.run import run
from pkg_resources import resource_filename

# Please define your values in here, option 1, and option 2.
jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
feature = 'PRDF'
feature_scaling = False
prop = 'formation_energy'
N_sample = 300
algorithm = 'PyTorch_ANN' # or dl

# Option 1: If you want to use an algorithm from Scikit-learn, please enter the following
level = 'light'
pipeline = False

# Option 2: If you want to use an algorithm from PyTorch, please enter the following
hidden_layers = {"n_layers": 1, "n_neurons": [10]}

# Running the user-defined values. Don't tresspass beyond this point.
runner = run(jsonfile=jsonfile, feature=feature, prop=prop, N_sample=N_sample)
runner.load_data()
runner.convert_data_1D() #choose cpu number if you want to active this function
runner.choose_feature(keys=feature) #choose feature combinations if you want
runner.ml_train(hidden_layers = hidden_layers)
runner.print_time()
