# REINFORCEMENT LEARNING PROJECT 
**Course:** Machine Learning & Deep Learning (MLDL) 2024/2025  
**Group ID:** 48

## Introduction
This project focuses on training a one-legged robot simulator to walk using Reinforce, Actor-Critic and PPO, and bridges the sim-to-real gap via domain randomization. 
Additionally, we experimentally evaluated an adversarial training curriculum on the Reinforce algorithm and DORAEMON’s dynamics curriculum on PPO, with the goal of further enhancing robustness.

## STRUCTURE

The repository is organized into folders, each of which contains a commented main notebook demonstrating how to use the implemented classes
and describing the operations performed. The modules and classes are commented to facilitate understanding of their functionality.


- **Reinforce_ActorCritic/**
  - **env/**
    - **assets/**
      - `hopper.xml` – xml file of the hopper environment
    - `custom_hopper.py` – custom hopper environment
    - `mujoco_env.py` – wrappers class for Mujoco environments
  - **models/** – trained models
  - **plots/** – saved plots
  - **reportFiles/** – folder containing csv files with training characteristics for each trained model
  - **results/** – folder containing txt files with actor-critic results
  - `classes.py` – classes definitions
  - `module_Reinforce_ActorCritic.py` – implementation of reinforce and actor-critic algorithms
  - `main_Reinforce_ActorCritic.ipynb` – main notebook containing graphs and experimentation code
 
- **PPO/**
  - **assets/**
    - `hopper.xml` – xml file of the hopper environment
  - **models/** – trained PPO models and tensorboard log
  - **plots/** – plots of some metrics of training and test with PPO
  - **PPOresults/**
    - `PPO_250515_12-46-20.monitor.csv` – csv file with results of PPO trainings
  - `custom_hopper.py` – custom hopper environment for PPO
  - `mujoco_env.py` – wrappers class for Mujoco environments
  - `main_PPO.ipynb` – main notebook containing trainings and tests of PPO models
  - `PPO_train_test.py` – class for training and evaluate PPO models

- **Doraemon/**
  - **env/**
    - **assets/**
      - `hopper.xml` – xml file of the hopper environment
    - `custom_hopper_doraemon.py` – custom hopper environment for doraemon
    - `mujoco_env.py` – wrappers class for Mujoco environments
  - **models/** – trained models with doraemon
  - `doraemon_module.py` – implementation of doraemon module
  - `main_doraemon.ipynb` – main notebook containing trainings with doraemon
  - `test_models.ipynb` – notebook containing tests of best models of doraemon

- **AdversarialTraining/**
  - **env/**
    - **assets/**
      - `hopper.xml` – xml file of the hopper environment
    - `custom_hopper.py` – custom hopper environment
    - `mujoco_env.py` – wrappers class for Mujoco environments
  - **models/** – trained models
  - **plots/** – saved plots
  - **reportFile/** – folder containing csv files with training characteristics for each trained model
  - `classes_beta_batch.py` – classes definitions
  - `module_Reinforce_ActorCritic_beta_batch.py` – implementation of reinforce and actor-critic algorithms with adversarial training features
  - `main_AdversarialTraining.ipynb` – main notebook containing demos and graphs

- `requirements.txt` – dependency list

# USAGE 
In this section, you’ll find where to find demos and how to run them.

## Prerequisites

First, install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Demo 

For each specified folder, open the specified notebook and run the specified cell.

- **Reinforce and Actor Critic**      
  Find a demo for reinforce in the **AdversarialTraining** folder.
  Notebook: `main_Reinforce_ActorCritic.ipynb`
  Refer to the notebook for previously run cells, plots and results (the notebook runs successfully, but it may take a long time to complete)

- **PPO**  
  Notebook: `main_PPO.ipynb`  
  Run the first two cells to load `PPOTrainer` and try the demo for training and evaluate a PPO model. 

- **Doraemon**  
  Notebook: `main_doraemon.ipynb`  
  Run the first cell to load `DomainRandDistribution` and `DORAEMON` and try the demo for training, evaluate and print results of the PPO model with Doraemon.

- **AdversarialTraining**  
  Notebook: `main_AdversarialTraining.ipynb`  
  Run the first cell to import modules and packages and run the _Train demo_ and _Test demo_ sections for results on reinforce algorithm with adversarial.


## AUTHORS
- Maddalena Ghiotti - s332834@studenti.polito.it 
- Letizia Greco - s336195@studenti.polito.it 
- Lorenzo Terna - s331113@studenti.polito.it
- Chiara Zambon - s329148@studenti.polito.it



