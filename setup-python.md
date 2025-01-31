# Setup instructions

## Local Development

- Create a virtual environment specific to the current project by running the following within the root of the project

```bash
# pyenv shell 3.10.12

# This will create a python virtual environment within the project in a new `.venv` folder.

# use this for mac
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
pip install -r requirements.txt

# use this for windows
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_dev.txt
pip install -r requirements.txt
```

```bash
# conversely, to exit the virtual environment
deactivate
```
