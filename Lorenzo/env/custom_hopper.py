"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
# CustomHopper è un ambiente personalizzato che estende quello Mujoco Standard al fine di
# supportare la randomizzazione del dominio e la manipolazione delle masse della gamba robotica

# Sostanzialmente questo file implementa un Wrapper dell'ambiente Hopper di Mujoco, ovvero un
# rivestimento che permette di modificare il comportamento dell'ambiente originale senza
# modificarne il codice sorgente. In questo caso, si possono modificare le masse
# della gamba robotica in modo casuale, in modo da rendere l'ambiente più robusto e
# generalizzabile. Inoltre, il wrapper implementa anche la funzionalità di salvataggio e
# caricamento dello stato dell'ambiente, in modo da poter riprendere l'addestramento da un
# determinato punto senza dover ricominciare da zero. Infine, esso implementa anche la
# funzionalità di visualizzazione dell'ambiente, in modo da poter vedere l'andamento
# dell'addestramento in tempo reale

from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # array of default link masses
                                                                        # si esclude lo 0 perchè di solito è il world body, statico

        if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift) (stiamo applicando una variazione sistematica al modello)
            self.sim.model.body_mass[1] *= 0.7
        # semplice estensione al caso if domain == 'target'
    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())


    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""
        
        #
        # TASK 6: implement domain randomization. Remember to sample new dynamics parameter
        #         at the start of each training episode.
        # ESEMPIO IMPLEMENTAZIONE: (da un'uniforme)

        # Variazioni del ±20% attorno alle masse originali
        variation_range = 0.2  
        lower = (1.0 - variation_range) * self.original_masses
        upper = (1.0 + variation_range) * self.original_masses

        sampled = self.np_random.uniform(low=lower, high=upper)
        return sampled


    def get_parameters(self):
        """Get value of mass for each link""" # utile per capire se si sta lavorando con il giusto ambiente (source o target)
        masses = np.array( self.sim.model.body_mass[1:] ) 
        return masses


    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt # il robot è premiato se avanza
        reward += alive_bonus # ricompensa costanta per essere vivo
        reward -= 1e-3 * np.square(a).sum() # penalità proporzionale al quadrato dell’azione --> disincentiva movimenti energici
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:], # posizione esclusa la x globale
            self.sim.data.qvel.flat      # tutte le velocità
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
# questo permette a Gym di riconoscere "CustomHopper-source-v0" come un ambiente valido
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

