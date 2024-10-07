# detecting-kidney-cancer

The overall goal of this project is to detect kidney cancer. This is done by training a model on around 10,000 images and the results of that can then be stored and evaluated. 

## Setup

```zsh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
pip install -r requirements.txt
```

## TODO
Fix the model so that it trains off the data properly \
\
Add sample data / csv that tests will automatically do \
\
Figure out how to make the training more acurate (maybe label the data / annotate the data) \
\
Use less epochs when training the data, maybe get a cpu/gpu in order to make the training better \
\
Do reinforcment learning on the data (create synthetic data that is accurate in order to train the model in a better way)
\
clean up the repo in general

## How to train and evaluate the model
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

## What the data is
The data contains around 10,000 example images of what a benign tumor is, cancerous tumor, kidney stone, and cystic kidney disease. Using all the data it is then possible to use the model to detect what is cancerous.

## References
Everything here is not necceasrily refrences I am using, it consists of both a brainstorm of potential helpful ideas and stuff I am actually using.

### Data
This is data of CT Scans that detected benign tumors, cancerous tumors, kidney stones, and cystic kidney disease.\
https://www.kaggle.com/datasets/anima890/kidney-ct-scan

### Outline
This is the general outline I currently have for the plan for CS Research.\
https://docs.google.com/document/d/17-IYpmrqE0HKDivBMHZIiV9BpNWg1-dO3UNp64TP-5A/edit#heading=h.lqkzgapjvm91

### Research
This is my document filled with all of my research that I have done for the project.\
https://docs.google.com/document/d/1jyCNkPNPi46cjxJ8s3Vau9Eh5qat1DngavqQPdU4fZU/edit#heading=h.ou0x58cx9yea

### Potential Data
I gathered a lot of potential data I can use here.\
https://docs.google.com/document/d/15bKcaG3rvsBdYztuoyvrZNVk1f7BVaxxSHiR8Exsnhw/edit#heading=h.2kxi2um9ycqv

### Research Paper Outline
https://docs.google.com/document/d/1NA_NqXS3aBjFXdRgnviKh4LZwhwwNswsaF5yXGZ71y8/edit
