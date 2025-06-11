import sys, os
# aggiungi la cartella due livelli sopra (il project root) nel path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))


import numpy as np
from agent import Policy
from custom_hopper_doraemon import make_hopper_with_params
from doraemon_trainer import DoraemonTrainer

def main():
    # Determine dims from a dummy env
    dummy = make_hopper_with_params(np.ones(5))
    state_dim = dummy.observation_space.shape[-1]
    act_dim   = dummy.action_space.shape[-1]

    policy   = Policy(state_dim, act_dim)
    # Initialize Beta(a,b) pairs for each mass parameter
    phi_init = np.array([[2.0, 2.0]] * dummy.original_masses.shape[0])
    trainer = DoraemonTrainer(
        policy=policy,
        env_factory=make_hopper_with_params,
        phi_init=phi_init,
        alpha_success=0.8,
        kl_eps=0.05,
        lr_phi=1e-3,
        batch_size=32,
        max_iter=1000,
        algorithm="reinforce"  # or "actor-critic"
    )
    trainer.train()

if __name__ == "__main__":
    main()
