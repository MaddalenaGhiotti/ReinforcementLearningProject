"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                - What is the action space in the Hopper environment? Is it discrete or continuous?
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                - When exactly is the episode over?
                - What is an action here?
"""
import pdb

import gym

from env.custom_hopper import *


def main():
	# Creo un ambiente simulato, variante personalizzata dell'Hopper
	env = gym.make('CustomHopper-source-v0') # questo enviroment è usato per il training
	# env = gym.make('CustomHopper-target-v0') # questo enviroment è usato per il test per capire se generalizza bene --> domain adaptation in RL (sim2real transfer)

	print('State space:', env.observation_space) # continue state-space
	print('Action space:', env.action_space) # continue action-space
	print('Dynamics parameters:', env.get_parameters()) # masses of each link of the Hopper e
                                                        # sono cruciali per capire come il robot interagisce con la fisica
														# e come variano i comportamenti tra "source" e "target"

	n_episodes = 500
	render = False # fa partire la visualizzazione grafica dell'ambiente
    # render = False # disabilita la visualizzazione grafica dell'ambiente

	for episode in range(n_episodes):
		done = False
		state = env.reset()	# Reset environment to initial state

		while not done:  # Until the episode is over
            # applica le forze ai giunti
			action = env.action_space.sample()	# Sample random action (es: [-0.3, 1.2, 0.0])
            # aggiorna la fisica tramite il simulatore MuJoCo
			# calcola il nuovo stato, la reward, valuta se l'agente è caduto (done si/no) e info opzionali (dettagli aggiuntivi tipo la posizione)
			state, reward, done, info = env.step(action)	# Step the simulator to the next timestep

			if render:
				env.render()

	

if __name__ == '__main__':
	main()
	

# possibili esplorazioni:
# - provare a cambiare il numero di episodi
# - loggare i reward e vedere se il reward medio aumenta o per capire quanto va male la random policy
# - stampare state e action per vedere come evolve il sistema
# - confrontare i reward tra source e target

#CIAO