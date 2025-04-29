import argparse
import numpy as np
import matplotlib.pyplot as plt


print('ciao a tutti')
print('ciao a tutti 3')
print('Ce la faremo?')

def plot_returns(return_array):
	
    plt.figure(figsize=(15,10))
    plt.title('RETURN')
    plt.plot(np.arange(0,args.n_episodes,args.plot_every), return_array)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid()
    #plt.savefig('Return',dpi=300)

    plt.figure(figsize=(15,10))
    plt.title('CUMULATIVE RETURN')
    plt.plot(np.arange(0,args.n_episodes,args.plot_every), np.cumsum(return_array))
    plt.xlabel('Episode')
    plt.ylabel('Cumulative return')
    plt.grid()
    #plt.savefig('CumulativeReturn',dpi=300)
	

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_episodes', default=10000, type=int, help='Number of training episodes')   ###############Changed from 100000
	parser.add_argument('--print_every', default=2000, type=int, help='Print info every <> episodes')   ###############Changed from 20000
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--plot', default=False, action='store_true', help='Plot the returns')
	parser.add_argument('--plot_every', default=50, type=int, help='Plot return every <> episodes')

	return parser.parse_args()

args = parse_args()



def main():
	array = np.array([2,4,7,3,4]*40)
	plot_returns(array)
	

if __name__ == '__main__':
	main()