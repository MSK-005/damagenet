# Damage Assessment Model
Given a pair of pre- and post-disaster satellite images, the model calculates the per-pixel damage factor across severity labels: no damage, minor damage, major damage, and destroyed. Developed on Python 3.14. The final model has 2 stages: the first stage is to teach the model to find houses by looking at the pre-disaster images. The second stage uses the stage 1 model as its own encoder, and performs the damage analysis.

## Dataset: xBD Dataset
The model uses the xBD Challenge dataset (found at [https://xview2.org/download](https://xview2.org/download)). Place the dataset in the `data/` folder.

## Setting Up the Project
After cloning the project, enter the following bash commands: 
```
pip install -r requirements.txt
pip install -e .
```

# Training
To use [WandB](https://wandb.ai/site) (Weights and Biases), create an account and get your API key. 

## Training on Kaggle
Open [Kaggle](https://www.kaggle.com), and import the appropriate notebook from the project's `notebooks/kaggle`. Use the "T4 x 2" accelerator option. In the "Secrets" section, create the WandB key by the name "WANDB_API_KEY" and set it to your API key. After training your respective model stage, you can download the model file from `models/stageX` (where X is 1 or 2).

## Training Locally
To use [WandB](https://wandb.ai/site) (Weights and Biases), create a `.env` at the project root and set the `WANDB_API_KEY` to your WandB API key.