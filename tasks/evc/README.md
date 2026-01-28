# Efficiency vs. coverage
This is a source code for Efficiency vs. coverage trade-off task in the paper.

## Usage

The source code is in the `main.py` file with these following command line arguments:
```bash
usage: main.py [-h]
               [--data DATA]
               [--num_data NUM_DATA]
               [--eta ETA]
               [--truncation TRUNCATION]
               [--dim DIM]
               [--num_trains NUM_TRAINS]
               [--num_cals NUM_CALS]
               [--num_cvs NUM_CVS]
               [--k_neighbors K_NEIGHBORS]
               [--file_path FILE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Dataset to use
  --num_data NUM_DATA   Number of data points
  --eta ETA             Number of samples each distribution
  --truncation TRUNCATION
                        Truncation level for Fourier coefficients
  --dim DIM             Dimension of the data
  --num_trains NUM_TRAINS
                        Number of training samples
  --num_cals NUM_CALS   Number of calibration samples
  --num_cvs NUM_CVS     Number of cross-validation samples
  --k_neighbors K_NEIGHBORS
                        Number of neighbors for adaptive CP
  --file_path FILE_PATH 
                        File path for loading data
```

Example command:
```bash
python -m tasks.evc.main --data "mixture_of_betas" --num_data 3200 --eta 1000 --truncation 10 --dim 1 --num_trains 1000 --num_cals 1000 --num_cvs 200 --k_neighbors 500
```
