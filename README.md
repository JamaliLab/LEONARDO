# LEONARDO
Learning Electron micrOscopy NAnoRod Diffusion via an attention netwOrk

## Authors

Zain Shabeeb, Naisargi Goyal, Pagnaa Attah Nantogmah, [Vida Jamali](https://vidajamali.github.io)


### Installation

- install [anaconda](https://docs.anaconda.com/anaconda/install/)
- conda env create -f LEONARDO.yml
- conda activate LEONARDO

  Download the LEONARDO model from Huggingface: https://huggingface.co/JamaliLab/LEONARDO
  Download the dataset from Huggingface: https://huggingface.co/datasets/JamaliLab/LEONDARDO
  Use the Jupyternotebook to generate new LPTEM trajectories using the pre-trained model or train a new model.

###  Abstract
The motion of nanoparticles in complex environments can provide us with a detailed understanding of interactions occurring at the molecular level. Liquid phase transmission electron microscopy (LPTEM) enables us to probe and capture the dynamic motion of nanoparticles directly in their native liquid environment, offering real time insights into nanoscale motion and interaction. However, linking the motion to interactions to decode underlying mechanisms of motion and interpret interactive forces at play is challenging, particularly when closed-form Langevin-based equations are not available to model the motion. Herein, we present LEONARDO, a deep generative model that leverages a physics-informed loss function and an attention-based transformer architecture to learn the stochastic motion of nanoparticles in LPTEM. We demonstrate that LEONARDO successfully captures statistical properties suggestive of the heterogeneity and viscoelasticity of the liquid cell environment surrounding the nanoparticles.

### Acknowledgements
This project is funded by the National Science Foundation Division of Chemical, Bioengineering, Environmental, and Transport Systems under award 2338466, and the American Chemical Society Petroleum Research Fund under award 67239-DNI5. 
