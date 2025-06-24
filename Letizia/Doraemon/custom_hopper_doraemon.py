import numpy as np
import gym
from gym import utils
from env.mujoco_env import MujocoEnv
import torch

import random

SEED = 42  
np.random.seed(SEED)
random.seed(SEED)

class CustomHopperDoraemon(MujocoEnv, utils.EzPickle):
    """Hopper environment tailored for the DORAEMON loop.

    Features
    --------
    * **Domain Randomization** – If an external `dr_distribution` (multivariate Beta) is provided,
      every episode samples a new vector of scaling factors for the 4 body masses and applies it.
    * **Episode Buffer** – Stores, at the end of each episode, a dictionary with the sampled
      dynamics parameters and the cumulative return so that DORAEMON can perform importance
      sampling. Retrieve it with `get_buffer()` and clear it with `reset_buffer()`.
    * **Train / Eval switch** – When `train_mode=False` the dynamics are kept fixed (useful for
      evaluation runs).
    """

    def __init__(self,
                 dr_distribution=None,
                 domain=None,
                 train_mode=True,
                 frame_skip: int = 4,
                 return_threshold: float = 500.0):
        
        # episode logging 
        self._episode_return = 0.0
        self._global_buffer = []     # list[dict]: {'dynamics': np.ndarray, 'return': float, 'success': float}

        self._current_scale = np.ones(4)  # current scaling factors for the 4 link masses
        self.return_threshold = return_threshold

        MujocoEnv.__init__(self, frame_skip)
        utils.EzPickle.__init__(self)

        # domain randomisation
        self.original_masses = np.copy(self.sim.model.body_mass[1:])  # 4 link masses
        self.dr_distribution = dr_distribution
        self.train_mode = train_mode

        # optional domain presets 
        if domain == 'source':       # lighter torso
            self.sim.model.body_mass[1] *= 0.7



    # Public helpers for DORAEMON

    def get_buffer(self):
        """Return the list of logged episodes collected so far."""
        buf = self._global_buffer.copy()  # return a copy to avoid external modifications
        self._global_buffer.clear()        # clear the global buffer after reading
        return buf

    def reset_buffer(self):
        """Erase the stored episode information (call after reading)."""
        self._global_buffer.clear() 

 
    # Domain-randomisation utilities

    def sample_parameters(self):
        """Sample a new set of randomized masses (shape: 3,) according to the DR distribution."""
        scale_factors = (self.dr_distribution.sample(1)[0]          
                        if self.dr_distribution is not None
                        else np.random.uniform(0.5, 1.5, size=self.original_masses[1:].shape))  # uniform in [0.5, 1.5] if no distribution is provided       
        masses = torch.tensor(self.original_masses[1:], dtype=scale_factors.dtype, device=scale_factors.device) * scale_factors
   
        self._current_scale = scale_factors             # meomrize the current scaling factors
        return masses

    def set_parameters(self, masses: np.ndarray):
        """Apply the given masses to the MuJoCo model."""
        self.sim.model.body_mass[2:] = masses


    # MuJoCoEnv API overrides

    def reset_model(self):
        """Called at every environment reset."""
        if self.train_mode:                          # sample new dynamics only in train mode
            self.set_parameters(self.sample_parameters())
        self._episode_return = 0.0                   # reset return tracker     

        # randomise initial state slightly
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """Return current observation vector (pos[1:], vel)."""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def step(self, action: np.ndarray):
        """Standard MuJoCo step with simple reward and termination criteria."""
        pos_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        pos_after, height, ang = self.sim.data.qpos[:3]

        #  reward 
        alive_bonus = 1.0
        reward = (pos_after - pos_before) / self.dt + alive_bonus    # forward progress + alive
        reward -= 1e-3 * np.square(action).sum()                     # control cost
        self._episode_return += reward

        #  termination
        done = not (np.isfinite(self.state_vector()).all() and height > 0.7 and abs(ang) < 0.2)

        info = {
            "episode": {
            "r": self._episode_return,
            "l": self.sim.data.time  
            },
                "is_success": float(self._episode_return >= self.return_threshold),
            }


        # buffer logging
        if done: 
            ep_rec = {
                "dynamics": np.copy(self._current_scale),
                "return":   self._episode_return,
                "success":  info["is_success"]
                }
            
            self._global_buffer.append(ep_rec)
            self._episode_return = 0.0

        return self._get_obs(), reward, done, info
    



# Gym registration

gym.envs.register(
    id="CustomHopperDoraemon-v0",
    entry_point=f"{__name__}:CustomHopperDoraemon",
    max_episode_steps=500
) 