# detecting-kidney-cancer

detects kidney cancer 

# Setup

```zsh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
pip install -r requirements.txt
```

# How to train and evaluate the model
There are several different ways to go about executing this code. Either run it all together or run it one by one. If you would like to evaluate the model and train it with one command run the following:
```zsh
make detect
```

## Details to train the model
In order to train the model run the following command:
```zsh
make train
```

## Details to evaluate the model
```zsh
make evaluate
```
# What the data means
The data contains example images of what a benign tumor is, cancerous tumor, kidney stone, and cystic kidney disease.

# References

## Data
This is data of CT Scans that detected benign tumors, cancerous tumors, kidney stones, and cystic kidney disease.
https://www.kaggle.com/datasets/anima890/kidney-ct-scan

