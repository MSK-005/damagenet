# Damage Assessment Model
Given a pair of pre- and post-disaster satellite images, the model calculates the per-pixel damage factor across severity labels: no damage, minor damage, major damage, and destroyed. Developed on Python 3.14.

## Dataset: xBD Dataset
The model uses the xBD Challenge dataset (found at [https://xview2.org/download](https://xview2.org/download)). Place the project in the `data/` folder.

## Setting Up the Project
After cloning the project, enter the following bash commands: 
```
pip install -r requirements.txt
pip install -e .
```

## Training
`python src/train.py`