# Ablation Study for Inference Time
This is a source code for Ablation Study for Inference Time task in the paper.

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

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Dataset to use
  --num_data NUM_DATA   Total number of data samples
  --eta ETA             Number of samples per distribution
  --truncation TRUNCATION 
                        Truncation level for Fourier coefficients
  --num_trains NUM_TRAINS
                        Number of training samples
  --num_cals NUM_CALS   Number of calibration samples
  --num_cvs NUM_CVS     Number of cross-validation samples

```

Example command:
```bash
python -m tasks.ablation_calibration_size.main \
  --data mixture_of_betas \
  --num_data 3200 \
  --eta 1000 \
  --truncation 10 \
  --num_trains 1000 \
  --num_cals 200000 \
  --num_cvs 200
```
