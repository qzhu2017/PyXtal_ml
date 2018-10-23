Scripts to download and covert the materials data from the public databases. 
In order to use the available functions in ML-Materials, we have customzied formats in json files.
- `formula`: string to present the formula, e.g. 'Al2O3'
- `lattice`: 3*3 matrix
- `coords`: an array to store the fractional coordinate
- `elements`: an array of elements e.g., ['Al', 'Al', ....]
- `band_gap`: band gap values
- `formation_energy`: normalized formation energy of this material in eV/atom

Currently, we provide the following scripts to convert data from
- Materials Project
- Jarvis-DFT
