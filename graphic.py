import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_alpha05 = 'actorcritic2_alpha0.5.csv'
csv_alpha025 = 'actorcritic2_alpha0.25.csv'
csv_alpha075 = 'actorcritic2_alpha0.75.csv'

loaded_alpha05 = pd.read_csv(csv_alpha05, index_col='model_name')
loaded_alpha025 = pd.read_csv(csv_alpha025, index_col='model_name')
loaded_alpha075 = pd.read_csv(csv_alpha075, index_col='model_name')



n_episodes=50_000
csvs=[ loaded_alpha025, loaded_alpha05, loaded_alpha075]
save_every= 100

type_names=['alpha=0.25','alpha=0.5', 'alpha=0.75']
colors=['darkseagreen','cornflowerblue','plum','darkseagreen','cornflowerblue','plum'] 
colors_dark=['green','blue','darkviolet','red','cyan','orange']

plt.figure(figsize=(12,10))
plt.title('ACTOR-CRITIC WITH DIFFERENT LOSS WEIGHTS - RETURNS DURING TRAINING ', fontsize=18)
for n_csv in range(len(csvs)):
    avg_type=np.zeros((20,))
    df = csvs[n_csv]
    sub_df = df[(df['n_episodes']==n_episodes)]
    for index, row in sub_df.iterrows():
        return_array = np.array(eval(row['returns']))
        avg_type+=return_array
        line = plt.plot(np.arange(len(return_array))*save_every, return_array, c=colors[n_csv], zorder=0,label='_nolegend_')
    line = plt.plot(np.arange(len(avg_type))*save_every, avg_type/len(sub_df), c=colors_dark[n_csv], zorder=2, label=f'b={type_names[n_csv]}')
ax = plt.gca()
ax.ticks_params(axis='both', which='major', labelsize=15)
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Return', fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.savefig('actorcritic2_alpha_comparison.png', dpi=300, bbox_inches='tight')
plt.show()