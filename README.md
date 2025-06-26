# REINFORCEMENT LEARNING PROJECT 
**Course:** Machine Learning & Deep Learning (MLDL) 2024/2025  
**Group ID:** 48

## Introduction
This project focuses on training a one-legged robot simulator to walk using Reinforce, Actor-Critic and PPO, and bridges the sim-to-real gap via domain randomization. 
Additionally, we experimentally evaluated an adversarial training curriculum on the Reinforce algorithm and DORAEMON’s dynamics curriculum on PPO, with the goal of further enhancing robustness.

## STRUCTURE

The repository is organized into folders, each of which contains a commented main notebook demonstrating how to use the implemented classes
and describing the operations performed. The modules and classes are commented to facilitate understanding of their functionality.


- **AdversarialTraining/**
  - **env/**
    - **assets/**
      - `hopper.xml` – xml file of the hopper environment
    - `custom_hopper.py` – custom hopper environment
    - `mujoco_env.py` – wrappers class for Mujoco environments
  - **models/** – (describe contents)
  - **plots/** – (describe contents)
  - **reportFile/** – (describe contents)
  - `agent.py` – (describe contents)
  - `classes.py` – (describe contents)
  - `maddalena_classes_beta_baseline.py` – (describe contents)
  - `maddalena_module_Reinforce_ActorCritic.py` – (describe contents)
  - `main_Maddalena.ipynb` – (describe contents)
  - `Piri_toRun.ipynb` – (describe contents)

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

- **Reinforce_ActorCritic/**
  - **env/**
    - **assets/**
      - `hopper.xml` – xml file of the hopper environment
    - `custom_hopper.py` – custom hopper environment
    - `mujoco_env.py` – wrappers class for Mujoco environments
  - **models/** – (describe contents)
  - **plots/** – (describe contents)
  - **reportFiles/** – (describe contents)
  - **results/** – (describe contents)
  - `classes.py` – (describe contents)
  - `main_Reinforce_ActorCritic.ipynb` – (describe contents)
  - `module_Reinforce_ActorCritic.py` – (describe contents)

- `requirements.txt` – dependency list

# USAGE 
In this section, you’ll find how to run the demos for the various classes.

## Prerequisites

First, install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Demo 

For each class, open the specified notebook and run the specified cell.

- **AdversarialTraining**  
  Notebook: `Piri_toRun.ipynb`  
  Run the ... cell to import `classname` and ...  

- **Doraemon**  
  Notebook: `main_doraemon.ipynb`  
  Run the first cell to load `DomainRandDistribution` and `DORAEMON` and try the demo for training, evaluate and print results of the PPO model with Doraemon.

- **PPO**  
  Notebook: `main_PPO.ipynb`  
  Run the first two cells to load `PPOTrainer` and try the demo for training and evaluate a PPO model. 

- **Reinforce and Actor Critic**      
  Notebook: `main_Reinforce_ActorCritic.ipynb`  
  Run the .... to load `classname` and ....


## AUTHORS
- Maddalena Ghiotti - s332834@studenti.polito.it 
- Letizia Greco - s336195@studenti.polito.it 
- Lorenzo Terna - s331113@studenti.polito.it
- Chiara Zambon - s329148@studenti.polito.it



