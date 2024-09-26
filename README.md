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
Training the model consists of compiling around 10,000 images worth of data that divided into 4 main categories: normal, tumor, stone, cyst. After training the model it should be able to know the difference between those 4 different categories. In order to train the model run the following command:
```zsh
make train
```

## Details to evaluate the model
The evaluation of the model bassically displays data to show how accurate the training was. It will create confusion matrices and show the overall percentages of the output. In order to train the model run the following command:
```zsh
make evaluate
```

# What the data is
The data contains around 10,000 example images of what a benign tumor is, cancerous tumor, kidney stone, and cystic kidney disease. Using all the data it is then possible to use the model to detect what is cancerous.

# References

## Data
This is data of CT Scans that detected benign tumors, cancerous tumors, kidney stones, and cystic kidney disease.
https://www.kaggle.com/datasets/anima890/kidney-ct-scan

