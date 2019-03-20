# cs234-final-project

## Before start

 - Make sure you are using python 3.6+
 - Make sure pip is installed with either virtualenv or not.
 
## Setup

 - Create a 'data' directory in the root of repo for your data and output.
 - Create a 'scores' directory under 'data' directory.
 - pip install -r requirments.txt

## Run

Run following command to preprocess dataset

```sh
python dataset.py
```

Run following command to run a LASSO bandit agent 10 times with default reward function

```sh
python main.py -a lasso -s 10 -r default
```
