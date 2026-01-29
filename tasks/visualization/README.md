# Visualization
This is a source code for Visualization task in the paper.

## Usage

The source code is in the `main.py` file with these following command line arguments:
```bash
usage: main.py [-h]
               [--data DATA]
               [--num_data NUM_DATA]
               [--eta ETA]
               [--truncation TRUNCATION]
               [--num_trains NUM_TRAINS]
               [--num_cals NUM_CALS]
               [--num_cvs NUM_CVS]
               [--file_path FILE_PATH]
               [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Dataset to use
  --num_data NUM_DATA   Number of data points
  --eta ETA             Number of samples each distribution
  --truncation TRUNCATION
                        Truncation level for Fourier coefficients
  --num_trains NUM_TRAINS
                        Number of training samples
  --num_cals NUM_CALS   Number of calibration samples
  --num_cvs NUM_CVS     Number of cross-validation samples
  --file_path FILE_PATH 
                        File path for loading data
  --seed SEED           Random seed
```

Example command:
```bash
python -m tasks.visualization.main --data "mixture_of_betas" --num_data 3200 --eta 1000 --truncation 10 --dim 1 --num_trains 1000 --num_cals 1000 --num_cvs 200
```

```bash
python -m tasks.visualization.main --data "mortality" --truncation 10 --dim 1 --num_trains 20 --num_cals 10 --num_cvs 2 --file_path data/Mortality.npy
```

```bash
python -m tasks.visualization.main --data "house_price" --truncation 10 --dim 1 --num_trains 200 --num_cals 100 --num_cvs 20 --file_path data/House_price_processed.npy
```