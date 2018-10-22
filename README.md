# ML-Materials
A library of ML training of materials' properties
- `datasets`: python class to download the data from open database + data in json format
- `descriptors`: python class for different types of descriptors (RDF, ADF, chemical labeling and enviroments)
- `ml`: python class for the choice of different pipelines of ML methods (KRR, ...)
- `test`: python class for unit test
- `results`: a collection of results from different data sources, descriptors and ML methods.

This code has the following hierarchy
```
root_dir
├── main.py
├── descriptors -> descriptors.py -> (RDF.py, ADF.py, DDF.py, Chem.py, .etc)
├── datasets -> collection.py -> (json files)
├── ml -> methods.py -> (KRR.py, KNN.py, .etc.)
├── test -> (various scripts to test the accuracy and efficiency of the code)
```
