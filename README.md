REINFORCEMENT LEARNING PROJECT 
MLDL course 24/25
group ID: 48

The project is organized into folders, each of which contains a commented main notebook demonstrating how to use the implemented classes
and describing the operations performed. The modules and classes are commented to facilitate understanding of their functionality.


## STRUCTURE

AdversarialTraining/
    env/
        assets/
            hopper.xml (xml file of the hopper environment)
        custom_hopper.py (costum hopper environment)
        mujoco_env.py (wrappers class for Mujoco environments)
    models/ (describe contents)
    plots/ (describe contents)
    reportFile/ (describe contents)
    agent.py (describe contents)
    classes.py (describe contents)
    maddalena_classes_beta_baseline.py (describe contents)
    maddalena_module_Reinforce_ActorCritic.py (describe contents)
    main_Maddalena.ipynb (describe contents)
    Piri_toRun.ipynb (describe contents)

Doraemon/
    doraemon_module.py (implementation of doraemon module)
    main_doraemon.ipynb (main notebook containing trainings with doraemon)
    test_models.ipynb (notebook containing tests of best models of doraemon)
    env/
        assets/
            hopper.xml (xml file of the hopper environment)
        custom_hopper_doraemon.py (custom hopper environment for doraemon)
        mujoco_env.py (wrappers class for Mujoco environments)
    models/ (trained models with doraemon)    
    

env/
    assets/
            hopper.xml (xml file of the hopper environment)
    custom_hopper.py (costum hopper environment)
    mujoco_env.py (wrappers class for Mujoco environments)

PPO/
    assets/
            hopper.xml (xml file of the hopper environment)
    models/ (trained PPO models and tensorboard log)
    plots/ (plots of some metrics of training and test with PPO)
    PPOresults/
        PPO_250515_12-46-20.monitor.csv (csv file with results of PPO trainings)
    custom_hopper.py (custom hopper environment for PPO)
    mujoco_env.py (wrappers class for Mujoco environments)
    main_PPO_train_test.ipynb (notebook containing trainings and tests of PPO models)
    PPO_train_test.py (class for training and evaluate PPO models)

Reinforce_ActorCritic/ 
    env/
        assets/
            hopper.xml (xml file of the hopper environment)
        custom_hopper.py (costum hopper environment)
        mujoco_env.py (wrappers class for Mujoco environments)
    models/ (describe contents)
    plots/ (describe contents)
    reportFiles/ (describe contents)
    results/ (describe contents)
    classes.py (describe contents)
    main_Reinforce_ActorCritic.ipynb (describe contents)
    module_Reinforce_ActorCritic.py (describe contents)

requirements.txt (dependency list)



