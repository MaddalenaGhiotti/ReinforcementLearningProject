import module_Reinforce_ActorCritic as rac
episodes = 5000
numbers, returns, average_returns, average_beginning, model_name = rac.train(0, hopper='S', n_episodes=episodes, trained_model=None, baseline=0, starting_threshold=700, save_every=75, print_every=episodes//5, plot=True, random_state=42, device='cpu')