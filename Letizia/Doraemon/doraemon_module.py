"""
Implementation of DORAEMON: Domain Randomization via Entropy Maximization.
Supports both PPO and SAC algorithms via Stable-Baselines3.
This version contiene implementazioni complete di:
  • Importance Sampling per il vincolo di performance
  • Ottimizzazione dell’entropia con aggiornamento reale dei parametri Beta
Assume che ogni CustomHopperDoraemon tenga un buffer di episodi in
self.episode_buffer = [{'dynamics': masses, 'return': R}, ...]
e che esponga env_method('get_buffer') che lo restituisce e poi lo resetti
con env_method('reset_buffer').
"""
import os
import numpy as np
import torch
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence
from scipy.optimize import minimize, NonlinearConstraint
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from custom_hopper_doraemon import CustomHopperDoraemon

MIN_PARAM, MAX_PARAM = 0.2, 5.0  # sigmoid bounds for beta parameters


# DomainRandDistribution

class DomainRandDistribution:
    """Multivariate Beta in [m, M] per dimension."""
    def __init__(self, dr_type: str, distr: list):
        assert dr_type == 'beta', 'Only beta supported'
        self.distr = distr.copy()
        self.ndims = len(distr)
        params = []
        for d in distr:
            params += [d['a'], d['b']]
        self.params = torch.tensor(params, dtype=torch.float32, requires_grad=True)
        self._build_distributions()

    # utilities
    def _build_distributions(self):
        self.to_distr = [Beta(self.params[2*i].clamp_min(0.05),self.params[2*i+1].clamp_min(0.05)) for i in range(self.ndims)]

    def get_stacked_bounds(self):
        return np.array([[d['m'], d['M']] for d in self.distr]).reshape(-1)

    def entropy(self):
        h = 0
        for i, d in enumerate(self.distr):
            m, M = d['m'], d['M'] 
            h += self.to_distr[i].entropy() + torch.log(torch.tensor(M-m))  # entropy H(X*(M-m)+m) = H(X) + log(M-m)
        return h
    
    def kl_divergence(self, other: "DomainRandDistribution") -> float:
        assert self.ndims == other.ndims, "KL divergence requires distributions with the same number of dimensions"
        kl = sum(kl_divergence(p, q) for p, q in zip(self.to_distr, other.to_distr))
        return float(kl)

    
 
    # Convenience helpers used by DORAEMON
 
    def get_stacked_params(self) -> np.ndarray:
        """Return the current Beta parameters as a flat numpy array
        [a1, b1, a2, b2, ...]  (length = 2 * ndims)."""
        return self.params.detach().numpy()


    # -------- sampling / pdf
    def sample(self, n_samples=1):
        vals = []
        for i, d in enumerate(self.distr):
            m, M = d['m'], d['M']
            x = self.to_distr[i].sample((n_samples,)).numpy()
            vals.append(x*(M-m)+m)
        return np.stack(vals, axis=1)

    def _univariate_pdf(self, value, i, log=False): # we need log for importance sampling
        d = self.distr[i]
        m, M = d['m'], d['M']
        # standardize in [0,1]
        z = (value-m)/(M-m)
        z = torch.as_tensor(z, dtype=self.params.dtype)
        if log:
            return self.to_distr[i].log_prob(torch.tensor(z)) - torch.log(torch.tensor(M-m))
        else:
            return torch.exp(self.to_distr[i].log_prob(torch.tensor(z)))/(M-m)

    def pdf(self, x, log=False):
        x = torch.as_tensor(x, dtype=self.params.dtype)
        if x.ndim == 1:          # (ndims,) → (1, ndims)
            x = x.unsqueeze(0)

        dens = 0 if log else 1
        for i in range(self.ndims):
            dens = dens + self._univariate_pdf(x[:, i], i, log=True) if log \
               else dens * self._univariate_pdf(x[:, i], i, log=False)
        return dens


    # ------------ helper static
    @staticmethod
    def sigmoid(x_opt, lb, ub): #for mapping parameters to [lb, ub] in optimization
        x_opt = torch.tensor(x_opt)
        return (ub-lb)/(1+torch.exp(-x_opt))+lb

    @staticmethod
    def inv_sigmoid(x, lb, ub):
        return -np.log((ub-lb)/(x-lb)-1)

    @staticmethod
    def beta_from_stacked(stacked_bounds, stacked_params): # build a domain-randomized Beta distribution candidate for DORAEMON
        distr = []
        for i in range(len(stacked_bounds)//2):
            d = {
                'm': stacked_bounds[2*i],
                'M': stacked_bounds[2*i+1],
                'a': stacked_params[2*i],
                'b': stacked_params[2*i+1]
            }
            distr.append(d)
        return DomainRandDistribution('beta', distr)


# TrainingSubRtn 

class TrainingSubRtn:
    """
    Training subroutine for DORAEMON.
    It handles the RL training using Stable-Baselines3, evaluates the policy,
    and manages the environment for domain randomization.
    """
    def __init__(self, env_id, dr_distribution, algo='PPO', lr=3e-4, gamma=0.99, seed=0,
                 n_eval_episodes=50, eval_freq=10000, return_threshold=500, device='cpu', run_path='.', verbose=0):
        self.env_id = env_id
        self.dr_distribution = dr_distribution
        self.algo = algo.upper()
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.n_eval_episodes = n_eval_episodes
        self.return_threshold = return_threshold
        self.eval_freq = eval_freq
        self.device = device
        self.run_path = run_path
        self.verbose = verbose
        self._env_ref = None  # keep reference to access buffer

    def _make_env_fn(self):
        return CustomHopperDoraemon(dr_distribution=self.dr_distribution, train_mode=True, return_threshold=self.return_threshold)

    def train(self, iter_idx, perf_thresh, max_steps, stopEarly=False):
        env = DummyVecEnv([self._make_env_fn]) # create a vectorized environment for Stable-Baselines3
        self._env_ref = env.envs[0]  # first (and only) env
        eval_env = CustomHopperDoraemon(dr_distribution=self.dr_distribution, train_mode=False, return_threshold=self.return_threshold)

        if self.algo == 'PPO':
            model = PPO('MlpPolicy', env, learning_rate=self.lr, gamma=self.gamma,
                        verbose=0, device=self.device, seed=self.seed)
        else:
            model = SAC('MlpPolicy', env, learning_rate=self.lr, gamma=self.gamma,
                        verbose=0, device=self.device, seed=self.seed)
        model.learn(total_timesteps=max_steps)

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True)
        model.save(os.path.join(self.run_path, f'model_iter{iter_idx}'))
        return mean_r, std_r, model.policy.state_dict(), model, max_steps

    #  buffer helpers
    def get_buffer(self):
        """Retrieve the buffer of dynamics and successes from the environment."""
        buf = self._env_ref.get_buffer()          
        dynamics  = np.array([b["dynamics"] for b in buf])
        successes = np.array([b["success"]  for b in buf])
        return dynamics, successes
    




# DORAEMON main loop 

class DORAEMON:
    def __init__(self, env_id, init_distr, target_distr, performance_lower_bound, return_threshold,
                 kl_upper_bound, algo='PPO', seed=0, budget=1_000_000, max_training_steps=100_000,
                 verbose=1):
        self.training_subrtn = TrainingSubRtn(env_id, init_distr, algo=algo, seed=seed, verbose=verbose, return_threshold=return_threshold)
        self.current_distr = init_distr
        self.performance_lower_bound = performance_lower_bound
        self.kl_upper_bound = kl_upper_bound
        self.return_threshold = return_threshold 
        self.budget = budget
        self.max_training_steps = max_training_steps
        self.verbose = verbose
        self.iteration = 0
        print(f"Init entropy: {self.current_distr.entropy().item():.3f}")

    def is_there_budget(self):
        return self.budget > 0

    def step(self):
        if not self.is_there_budget():
            return False

        # RL training
        mean_r, std_r, _, _, used_ts = self.training_subrtn.train(self.iteration, self.performance_lower_bound,
                                                                  min(self.budget, self.max_training_steps))
        self.budget -= used_ts
        print(f"Iter {self.iteration}: reward {mean_r:.1f} ± {std_r:.1f}  budget {self.budget}")

        # gather buffer (dynamics, returns)
        dynamics, successes = self.training_subrtn.get_buffer()
        dyn = torch.as_tensor(dynamics, dtype=self.current_distr.params.dtype)
        succ = torch.as_tensor(successes,  dtype=self.current_distr.params.dtype)


        # build optimisation elements
        stacked_bounds = self.current_distr.get_stacked_bounds()

        # perfrormance constraint: importance sampling
        def perf_fn(x_opt):
            beta_params = DomainRandDistribution.sigmoid(x_opt, MIN_PARAM, MAX_PARAM)
            proposed = DomainRandDistribution.beta_from_stacked(stacked_bounds, beta_params)
            w_log = proposed.pdf(dyn, log=True) - self.current_distr.pdf(dyn, log=True)
            w = torch.exp(w_log)
            est = torch.mean(w * succ).item()
            return est

        # check if performance constraint is feasible
        perf_cons = NonlinearConstraint(perf_fn, lb=self.performance_lower_bound, ub=np.inf)

        #  KL divergence constraint 
        def kl_fn(x_opt):
            beta_params = DomainRandDistribution.sigmoid(x_opt, MIN_PARAM, MAX_PARAM).detach().numpy()
            cand = DomainRandDistribution.beta_from_stacked(stacked_bounds, beta_params)
            return self.current_distr.kl_divergence(cand)
        
        # KL divergence constraint: KL(φ_prev || φ_cand) <= ε
        kl_cons = NonlinearConstraint(fun=kl_fn,
                              lb=-np.inf,
                              ub=self.kl_upper_bound)  

        # entropy maximization (-entropy minimization)
        def objective(x_opt):
            beta_params = DomainRandDistribution.sigmoid(x_opt, MIN_PARAM, MAX_PARAM).detach().numpy()
            proposed = DomainRandDistribution.beta_from_stacked(stacked_bounds, beta_params)
            return -proposed.entropy().item()
        
        # starting point for optimization
        x0 = DomainRandDistribution.inv_sigmoid(self.current_distr.get_stacked_params(), MIN_PARAM, MAX_PARAM)

        enforce_perf = perf_fn(x0) >= self.performance_lower_bound

        if not enforce_perf:
           
            res_try = minimize(
            fun=lambda x: -perf_fn(x),       # maximize performance
            x0=x0,
            method='trust-constr',
            constraints=[kl_cons],           # only KL constraint
            options={'xtol':1e-6,'gtol':1e-4,'maxiter':50}
            )

            if perf_fn(res_try.x) >= self.performance_lower_bound:
            # we got to a feasible point under KL constraint, use it as starting point
                x0 = res_try.x
                enforce_perf = True
                if self.verbose:
                    print("Backup distribution found: it satisfies performance bound.")
            else:
            # performance bound not satisfied, but we maximized it under KL constraint
                x0 = res_try.x
                if self.verbose:
                    print("Backup distribution found, but it does not satisfy performance bound.")
        # final constraints
        constraints = [kl_cons, perf_cons] if enforce_perf else [kl_cons]
        fun_objective = objective if enforce_perf else lambda x: -perf_fn(x)         

        # run scipy optimiser 
        res = minimize(
            fun=fun_objective,
            x0=x0,
            method='trust-constr',
            constraints=constraints,
            options={'xtol': 1e-6, 'gtol': 1e-4, 'maxiter': 100}
        )

        ok = res.success and (perf_fn(res.x) >= self.performance_lower_bound if enforce_perf else True)

        if not ok:
            if self.verbose:
                print("Fallback: max performance under KL constraint.")
            
            res = minimize(
                fun=lambda x: -perf_fn(x),
                x0=x0,
                method='trust-constr',
                constraints=[kl_cons],
                options={'xtol':1e-6, 'gtol':1e-4, 'maxiter':100}
            )

        # update distribution with solution (if feasible)
        opt_params = DomainRandDistribution.sigmoid(res.x, MIN_PARAM, MAX_PARAM).detach().numpy()
        with torch.no_grad():
            self.current_distr.params.copy_(torch.tensor(opt_params))
            self.current_distr._build_distributions()

        print(f"Iter {self.iteration}: entropy {self.current_distr.entropy():.3f}, perf {perf_fn(res.x):.3f}, KL {kl_fn(res.x):.3f}")
        self.iteration += 1
        return True

    def run(self):
        while self.step():
            pass
        print("DORAEMON finished. Final entropy:", self.current_distr.entropy().item())

