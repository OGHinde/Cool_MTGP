# Cool_MTGP

## A CONDITIONAL ONE-OUTPUT LIKELIHOOD FORMULATION FOR MULTITASK GAUSSIAN PROCESSES

Authors: 
  Óscar García Hinde (oghinde@tsc.uc3m.es)
  Vanessa Gómez Verdejo (vanessa@tsc.uc3m.es)
  Manel Martínez Ramón (manel@unm.edu)

This repository provides libraries and code to use the model proposed in the original paper, as well as replicate most of the results.

The repository is still very much a work in progress.

The datasets in /datasets/real/ were downloaded from http://mulan.sourceforge.net/datasets-mtr.html

WHAT WORKS:
*Please check out the CoolMTGP_basic_demo and CoolMTGP_variance_demo notebooks, which showcase the model's basic functionalities.
*To run the real dataset experiments, simply execute the run_experiments_real_cooltorch.py script with the dataset name (eg. 'andro') and kernel type ('Linear'       or 'RBF) as input arguments.

TO DO:
  - The /lib/ folder needs to be cleaned and organised.
  - The full pipeline for the synthetic experiments needs to be uploaded.
  - A user-friendly version for the PyTorch version of the model needs to be implemented.
