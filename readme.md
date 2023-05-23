# Latent Flow Matching

## Installation
Python 3.8 and Pytorch 1.13.1 are used in this implementation.
Please install required libraries:
```
pip install -r requirements.txt
```

## Training
All training scripts are wrapped in `run.sh`. Simply comment/uncomment the relevant commands and run `bash run.sh`.

## Testing
Some pieces of test scripts are included in `run_test.sh`. Following the same procedure as [training above](#training).

For massive testing on various epochs, please first modify some arguments in [test_laflo_slurm.py](./test_laflo_slurm.py) and then run `python test_laflo_slurm.py` to automatically generate bash script.


