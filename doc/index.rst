.. PyXtal_ML documentation master file, created by
   sphinx-quickstart on Tue Nov 13 15:03:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyXtal_ML's documentation!
=====================================

Introduction
==================================

    PyXtal_ml is an open source Python library for machine learning of crystal properties and force field. It is available for use under the MIT license. A basic tutorial is provided below for common functions. Additionally, documentation and source code are provided for individual modules. For more information about the project's development, see the GitHub page: https://github.com/qzhu2017/PyXtal_ml

Dependencies
============

Versions indicated are those used during development. Other versions may be compatible, but have not yet been tested.

  * `SciPy 1.0.1 <https://www.scipy.org/install.html>`_  
  * `NumPy 1.14.3 <https://www.scipy.org/scipylib/download.html>`_  
  * `Pymatgen 2017.9.3 <http://pymatgen.org/#getting-pymatgen>`_
  * `Scikit-learn  <http://scikit-learn.org/stable/>`_
  * `Keras  <https://keras.io/>`_
  * `Pytorch <https://pytorch.org/>`_


Installation
============

To install PyXtal_ml, first install all dependencies, then make a copy of the source code:

``git clone https://github.com/qzhu2017/pyxtal_ml``

Then, inside of the downloaded directory, run

``python setup.py install``

This will install the module. The code can be used within Python via

.. code-block:: Python

   import pyxtal_ml

Quick Start
============

.. code-block:: Python
  
   from pyxtal_ml.run import run
   from pkg_resources import resource_filename

   # Please define your values in here
   jsonfile = resource_filename("pyxtal_ml", "datasets/nonmetal_MP_8049.json")
   feature = 'Chem' 
   feature_scaling = 'MinMaxScaler'
   prop = 'formation_energy'
   N_sample = 300
   library = 'SkLearn' # SkLearn or pytorch
   algorithm = 'KRR' # or dl

   # Option 1: If you want to use an algorithm from Scikit-learn, please enter the following
   level = 'light'
   pipeline = False

   # Option 2: If you want to use an algorithm from PyTorch, please enter the following
   hidden_layers = {"n_layers": 3, "n_neurons": [50]}

   # Running the user-defined values. Don't tresspass beyond this point.
   runner = run(jsonfile=jsonfile, feature=feature, prop=prop, N_sample=N_sample, library=library,
            algo=algorithm, feature_scaling=feature_scaling, level=level, 
            pipeline=pipeline, hidden_layers=hidden_layers)
   runner.load_data()
   runner.convert_data_1D() #choose cpu number if you want to active this function
   runner.choose_feature(keys=feature) #choose feature combinations if you want
   runner.ml_train(algo=algorithm)
   runner.print_time()


The current version is 0.1dev. Expect frequent updates.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
