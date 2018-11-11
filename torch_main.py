from pyxtal_ml.run_torch import run
from pkg_resources import resource_filename

jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
N_sample = 300
feature = 'PRDF'
prop = 'formation_energy'
hidden_layers = {"n_layers": 1, "n_neurons": [10]}

runner = run(jsonfile=jsonfile, feature=feature, prop = prop, N_sample=N_sample)
runner.load_data()
runner.convert_data_1D() #choose cpu number if you want to active this function
runner.choose_feature(keys='PRDF') #choose feature combinations if you want
runner.ml_train(hidden_layers = hidden_layers)
runner.print_time()