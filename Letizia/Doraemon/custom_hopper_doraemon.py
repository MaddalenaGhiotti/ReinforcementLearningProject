import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv

def make_hopper_with_params(xi: np.ndarray):
    """Factory function: creates a CustomHopper environment with masses set by xi."""
    env = CustomHopper(train_mode=True)
    env.set_parameters(xi)
    return env

class CustomHopper(MujocoEnv, utils.EzPickle):
    """
    Hopper environment with support for controlled domain randomization.
    Parameter sampling is handled externally by DoraemonTrainer via set_parameters.
    """
    def __init__(self, train_mode=True):
        # Initialize the MuJoCo environment (4-frame skip)
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        # Store the original masses of all body links (excluding the base)
        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        self.train_mode = train_mode

    def set_parameters(self, xi: np.ndarray):
        """
        Set the body link masses according to the provided xi vector.

        Args:
            xi (np.ndarray): Array of masses matching original_masses shape.
        """
        assert xi.shape == self.original_masses.shape, \
            f"Shape of xi {xi.shape} is not compatible with original masses {self.original_masses.shape}."
        # Assign new masses and update the simulation model
        self.sim.model.body_mass[1:] = xi
        self.sim.forward()

    def reset_model(self):
        """
        Reset the simulation state without altering the masses.
        Mass values must be set via set_parameters before calling reset.
        """
        # Randomize initial position and velocity slightly
        qpos = self.init_qpos + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-.005, high=.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        """
        Perform one simulation step with the given action.

        Returns:
            ob (np.ndarray): Observation after the step.
            reward (float): Computed reward for this transition.
            done (bool): Whether the episode has terminated.
            info (dict): Additional information including 'success' flag.
        """
        # Measure forward progress
        pos_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        pos_after, height, angle = self.sim.data.qpos[0:3]

        # Compute reward: forward progress + alive bonus - control cost
        alive_bonus = 1.0
        reward = (pos_after - pos_before) / self.dt + alive_bonus
        reward -= 1e-3 * np.square(action).sum()

        # Check termination conditions
        state_vec = self.state_vector()
        done = not (
            np.isfinite(state_vec).all()
            and (np.abs(state_vec[2:]) < 100).all()
            and (height > 0.7)
            and (abs(angle) < 0.2)
        )

        ob = self._get_obs()
        # Indicate success if the episode did not terminate prematurely
        return ob, reward, done, {"success": not done}

    def _get_obs(self):
        """
        Return the current observation consisting of positions (excluding root) and velocities.
        """
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

# Register the environment with Gym
gym.envs.register(
    id="CustomHopper-DORAEMON-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
)
