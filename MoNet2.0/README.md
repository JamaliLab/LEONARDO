# MoNet2.0
Dilated CNN classifier for anomlous diffusion models

## Installation
- First create a conda enviornment for MoNet2.0 with python 3.10.14: `conda create -n monet2 python=3.10.14`

- Then install the requirements: `pip install -r requirements.txt`

## Usage
- `run_monet2_classification.py` runs the trained MoNet2.0 to generate the confusion matrix for the model showing the model classification performance, and a bar chart showing the percentage of LEONARDO-Generated trajectories classified as different diffusion classes.

- `run_monet2_fd.py` runs the trained MoNet2.0 to generate a lower triangular matrix of Frechet distance values. Each cell in the matrix represents the Frechet distance between the second-last layer of MoNet2.0 with input trajectories from different diffusion classes.

- A trained MoNet2.0 model is present in the repository as `monet2.keras`. To train MoNet2.0 from scratch, use the `monet_train.py` file. Download the training dataset from Huggingface: [https://huggingface.co/datasets/JamaliLab/MoNet2.0](https://huggingface.co/datasets/JamaliLab/MoNet2.0). These datasets comprise particle motion trajectories from the following diffusion classes:
  - Brownian Motion (BM)
  - Fractional Brownian Motion (FBM)
  - Continuous Time Random Walk (CTRW)
  - Annealed Transient Time Motion (ATTM)
  - Scaled Brownian Motion (SBM)
  - Lévy Walk (LW)
  - Real experimental LPTEM trajectories of gold nanorods diffusing in water

  FBM, CTRW, ATTM, SBM, and LW were simulated using the models from the Anomalous Diffusion (AnDi) challenge ([https://github.com/AnDiChallenge](https://github.com/AnDiChallenge)) without additional noise

For generating and analyzing particle motion trajectories using LEONARDO, check out [https://github.com/JamaliLab/LEONARDO](https://github.com/JamaliLab/LEONARDO).
