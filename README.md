# LEONARDO ([https://doi.org/10.1038/s41467-025-61632-1](https://doi.org/10.1038/s41467-025-61632-1))
Learning Electron micrOscopy NAnoRod Diffusion via an attention netwOrk

## Authors

Zain Shabeeb, Naisargi Goyal, Pagnaa Attah Nantogmah, [Vida Jamali](https://vidajamali.github.io)

## LEONARDO Description 
LEONARDO is a generative model with an attention-based transformer variational autoencoder architecture and a physics-informed loss function.

### Installation

- install [anaconda](https://docs.anaconda.com/anaconda/install/)
- `conda env create -f LEONARDO.yml`
- `conda activate LEONARDO`

Download the LEONARDO model from Huggingface: [https://huggingface.co/JamaliLab/LEONARDO](https://huggingface.co/JamaliLab/LEONARDO)
  
Download the dataset from Huggingface: [https://huggingface.co/datasets/JamaliLab/LEONDARDO](https://huggingface.co/datasets/JamaliLab/LEONDARDO)
  
Use the [LEONARDO_introductory_notebook.ipynb](https://github.com/JamaliLab/LEONARDO/blob/main/LEONARDO_introductory_notebook.ipynb) to generate new LPTEM trajectories using the pre-trained model or train a new model.

For using MoNet2.0 model checkout [https://github.com/JamaliLab/MoNet2.0](https://github.com/JamaliLab/MoNet2.0).

###  Abstract
The motion of nanoparticles in complex environments can provide us with a detailed understanding of interactions occurring at the molecular level. Liquid phase transmission electron microscopy (LPTEM) enables us to probe and capture the dynamic motion of nanoparticles directly in their native liquid environment, offering real time insights into nanoscale motion and interaction. However, linking the motion to interactions to decode underlying mechanisms of motion and interpret interactive forces at play is challenging, particularly when closed-form Langevin-based equations are not available to model the motion. Herein, we present LEONARDO, a deep generative model that leverages a physics-informed loss function and an attention-based transformer architecture to learn the stochastic motion of nanoparticles in LPTEM. We demonstrate that LEONARDO successfully captures statistical properties suggestive of the heterogeneity and viscoelasticity of the liquid cell environment surrounding the nanoparticles.

### Acknowledgements
This project is funded by the National Science Foundation Division of Chemical, Bioengineering, Environmental, and Transport Systems under award 2338466, the American Chemical Society Petroleum Research Fund under award 67239-DNI5, and the Georgia Tech Institute for Matter and Systems, Exponential Electronics seed grant.
### Citation
If you are using this code, please reference our paper:
```
  @article{shabeeb2025learning,
  title={Learning the diffusion of nanoparticles in liquid phase TEM via physics-informed generative AI},
  author={Shabeeb, Zain and Goyal, Naisargi and Attah Nantogmah, Pagnaa and Jamali, Vida},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={6298},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```    
