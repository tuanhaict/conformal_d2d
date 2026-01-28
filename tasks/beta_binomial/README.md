# Beta-binomial verification
This is a source code for Beta-binomial verification task in the paper.

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
               [--model MODEL]
               [--num_tests NUM_TESTS]
               [--num_iter NUM_ITER]
               [--file_path FILE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Dataset to use
  --num_data NUM_DATA   Total number of data samples
  --eta ETA             Number of samples per distribution
  --truncation TRUNCATION
                        Truncation level for Fourier coefficients
  --dim DIM             Dimension of the data
  --num_trains NUM_TRAINS
                        Number of training samples
  --num_cals NUM_CALS   Number of calibration samples
  --num_cvs NUM_CVS     Number of cross-validation samples
  --k_neighbors K_NEIGHBORS
                        Number of neighbors for adaptive CP
  --model MODEL         Model type: ot_map, nonparametric,
                        wasserstein_regression
  --num_tests NUM_TESTS
                        Number of test samples
  --file_path FILE_PATH 
                        File path for loading data
  --num_iter NUM_ITER   Number of iterations for calculating empirical coverage


```

Example command:
```bash
python -m tasks.beta_binomial.main \
  --data mixture_of_betas \
  --num_data 6200 \
  --eta 1000 \
  --truncation 10 \
  --dim 1 \
  --num_trains 1000 \
  --num_cals 1000 \
  --num_cvs 200 \
  --k_neighbors 500 \
  --model ot_map \
  --num_tests 1000 \
  --num_iter 50000
```
