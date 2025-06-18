"""
Implementation of DORAEMON: Domain Randomization via Entropy Maximization.
Supports both PPO algorithm via Stable-Baselines3.

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
from stable_baselines3.common.monitor import Monitor
from typing import List


import random

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

from custom_hopper_doraemon import CustomHopperDoraemon

MIN_PARAM, MAX_PARAM = 0.05, 30.0 # min and max values for Beta parameters 


# DomainRandDistribution

class DomainRandDistribution:
    """Multivariate Beta on dimension‑specific intervals."""

    def __init__(self, distr: List[dict]):  # each dict: {m,M,a,b}
        self.distr = distr
        self.ndims = len(distr)
        self.params = torch.tensor([v for d in distr for v in (d["a"], d["b"])], dtype=torch.float32, requires_grad=True)
        self._build()

    def _build(self): 
        """Build Beta distributions from parameters."""
        self.to_distr = [
            Beta(self.params[2 * i].clamp_min(0.05), self.params[2 * i + 1].clamp_min(0.05))
            for i in range(self.ndims)
        ]

    # ---------------- entropy / KL ---------------------------------------
    def entropy(self) -> float:
        """Return scalar entropy of the distribution (as float)."""
        h = 0.0
        for i, d in enumerate(self.distr):
            m, M = d["m"], d["M"]
            h += self.to_distr[i].entropy().item() + np.log(M - m)
        return h


    def kl_divergence(self, other: "DomainRandDistribution") -> float:
        """Compute the KL divergence between two distributions."""
        return float(sum(kl_divergence(p, q) for p, q in zip(self.to_distr, other.to_distr)))
    
        # -------- helper: stacked bounds ------------------------------------
    def get_stacked_bounds(self) -> np.ndarray:
        """
        Return [m1, M1, m2, M2, ...]  (shape = 2*ndims)
        useful for constructing the Beta distribution from parameters.
        """
        return np.array([[d["m"], d["M"]] for d in self.distr]).ravel()


    # ---------------- sampling / pdf -------------------------------------
    def sample(self, n: int = 1) -> torch.Tensor:
        """Sample from the multivariate distribution."""
        vals = []
        for i, d in enumerate(self.distr):
            m, M = d["m"], d["M"]
            x = self.to_distr[i].sample((n,))
            vals.append(x * (M - m) + m)
        return torch.stack(vals, dim=1)

    def _uni_pdf(self, x, i: int, log=False):
        """Compute the PDF of the i-th Beta distribution."""
        m, M = self.distr[i]["m"], self.distr[i]["M"]
        z = (torch.as_tensor(x) - m) / (M - m)
        if log:
            return self.to_distr[i].log_prob(z) - torch.log(torch.tensor(M - m))
        return torch.exp(self.to_distr[i].log_prob(z)) / (M - m)

    def pdf(self, x, *, log=False):
        """Compute the PDF of the multivariate distribution."""
        x = torch.as_tensor(x, dtype=self.params.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        res = 0.0 if log else 1.0
        for i in range(self.ndims):
            res = res + self._uni_pdf(x[:, i], i, log=True) if log else res * self._uni_pdf(x[:, i], i)
        return res

    # ---------------- helpers --------------------------------------------
    @staticmethod
    def sigmoid_param(x, MIN_PARAM, MAX_PARAM) -> np.ndarray:
        """Transform x to a sigmoid function in [MIN_PARAM, MAX_PARAM]."""
        return MIN_PARAM + (MAX_PARAM - MIN_PARAM) / (1 + np.exp(-x))
    
    @staticmethod
    def inverse_sigmoid_param(x, MIN_PARAM, MAX_PARAM) -> np.ndarray:
        """Inverse of sigmoid_param."""
        return -np.log((MAX_PARAM - MIN_PARAM) / (x - MIN_PARAM) - 1)


    @staticmethod
    def beta_from_stacked(bounds: np.ndarray, params: np.ndarray) -> "DomainRandDistribution":
        """ Create a DomainRandDistribution from stacked bounds and parameters. """
        d = []
        for i in range(len(bounds) // 2):
            d.append({"m": bounds[2 * i], "M": bounds[2 * i + 1], "a": params[2 * i], "b": params[2 * i + 1]})
        return DomainRandDistribution(d)
    

    # ------------- quick helper ----------------------------------------
    def as_numpy(self) -> np.ndarray:
        """Return current Beta parameters as a detached NumPy array."""
        return self.params.detach().numpy()


# TrainingSubRtn 

class TrainingSubRtn:
    """Subroutine for training on the source domain with PPO."""
    def __init__(
        self,
        dr_distribution: DomainRandDistribution,
        lr=1e-3,
        seed=42,
        n_eval_episodes=50,
        return_threshold=500.0,
        device="cpu",
        run_path="."
    ):

        self.dr_distribution = dr_distribution
        self.n_eval_episodes = n_eval_episodes
        self.return_threshold = return_threshold
        self.device = device
        self.run_path = run_path
        
        env = CustomHopperDoraemon(
            dr_distribution=self.dr_distribution,
            train_mode=True,
            return_threshold=return_threshold,
            domain="source"
            )
        
        env.seed(GLOBAL_SEED)
        self.vec_env = DummyVecEnv([lambda: env])

        self._env_ref = env  # reference to the environment for buffer access   



        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=lr,
            n_steps=4096,          
            batch_size=256,
            n_epochs=10,
            verbose=0,
            seed=seed,
            device=device,       
        )      

        
    # -------------------------------- train & evaluate on source
    def train(self, iter_idx: int, max_steps: int) -> tuple:
        self.model.learn(total_timesteps=max_steps, reset_num_timesteps=False)
        eval_env = CustomHopperDoraemon(
                dr_distribution=self.dr_distribution,
                train_mode=False,
                return_threshold=self.return_threshold,
                domain="source",
            )
        eval_env.seed(GLOBAL_SEED)
        eval_env = Monitor(eval_env)

        mean, std = evaluate_policy(self.model, eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True)
        self.model.save(os.path.join(self.run_path, f"model_iter{iter_idx}"))
        return mean, std, max_steps

    def get_buffer(self):
        """Get the buffer from the environment reference."""
        buf = self._env_ref.get_buffer(); self._env_ref.reset_buffer()
        dyn = np.array([b["dynamics"] for b in buf]); succ = np.array([b["success"] for b in buf])
        return dyn, succ
    




# DORAEMON main loop 

class DORAEMON:
    """DORAEMON: Domain Randomization via Entropy Maximization."""
    def __init__(
        self,
        *,
        init_distr: DomainRandDistribution,
        performance_lower_bound: float,
        return_threshold: float,
        kl_upper_bound: float,
        seed: int = 42,
        budget: int = 1_000_000,
        max_training_steps: int = 100_000,
        verbose: int = 1,
    ):
        self.train_sub = TrainingSubRtn(
            init_distr,
            seed=seed,
            return_threshold=return_threshold,
        )
        self.current = init_distr
        self.perf_lb = performance_lower_bound
        self.kl_ub = kl_upper_bound
        self.return_threshold = return_threshold
        self.budget = budget
        self.max_ts = max_training_steps
        self.verbose = verbose
        self.iter = 0

        if self.verbose:
            print(f"Init entropy: {self.current.entropy():.3f}")

    # --------------------------------------------------------------
    def _update_distribution(self, dyn: torch.Tensor, succ: torch.Tensor):
        """Solve the constrained optimisation to update Beta params."""

        bounds = self.current.get_stacked_bounds()

        @torch.no_grad()
        def _build_candidate(self, x_opt_np: np.ndarray) -> DomainRandDistribution:
            """Build a candidate distribution from the optimised parameters."""
            beta_par = torch.tensor(
            DomainRandDistribution.sigmoid_param(x_opt_np, MIN_PARAM, MAX_PARAM),
            dtype=self.current.params.dtype,
            device=self.current.params.device)
            return DomainRandDistribution.beta_from_stacked(bounds, beta_par)

        # ----- performance constraint (importance sampling) ----------
        def perf_fn(x_opt):
            cand = _build_candidate(self, x_opt)
            w_log = cand.pdf(dyn, log=True) - self.current.pdf(dyn, log=True)
            return float(torch.mean(torch.exp(w_log) * succ))

        perf_cons = NonlinearConstraint(perf_fn, lb=self.perf_lb, ub=np.inf)

        # ----- KL constraint ----------------------------------------
        def kl_fn(x_opt):
            cand = _build_candidate(self, x_opt)
            return self.current.kl_divergence(cand)

        kl_cons = NonlinearConstraint(kl_fn, lb=-np.inf, ub=self.kl_ub)

        # ----- objective: maximise entropy (= minimise -entropy) -----
        def objective(x_opt):
            cand = _build_candidate(self, x_opt)
            return -cand.entropy()

        # ----- optimisation start point -----------------------------
        x0 = self.current.params.detach().numpy()
        x0 = DomainRandDistribution.inverse_sigmoid_param(x0, MIN_PARAM, MAX_PARAM)

        res = minimize(fun=objective,
                x0=x0,
               method="trust-constr",
               constraints=[perf_cons, kl_cons],
               options={"maxiter": 200, "xtol": 1e-6, "gtol": 1e-4})

        
        # fallback: maximise perf under KL if entropy solve failed
        if not res.success:
            res = minimize(
                fun=lambda x: -perf_fn(x),
                x0=x0,
                method="trust-constr",
                constraints=[kl_cons],
                options={"xtol": 1e-6, "gtol": 1e-4, "maxiter": 200},
            )

        # update current distribution
        new_params = DomainRandDistribution.sigmoid_param(res.x, MIN_PARAM, MAX_PARAM)
        with torch.no_grad():
            self.current.params.copy_(torch.tensor(new_params))
            self.current._build()

        return perf_fn(res.x), kl_fn(res.x)

    # --------------------------------------------------------------
    def step(self) -> bool:
        """One DORAEMON iteration: train → optimise → log → eval on target."""
        if self.budget <= 0:
            return False

        # ----- RL training on source --------------------------------
        max_steps = min(self.budget, self.max_ts)
        mean_src, std_src, used_ts = self.train_sub.train(self.iter, max_steps)
        self.budget -= used_ts

        if self.verbose:
            print(f"Iter {self.iter}: source R {mean_src:.1f} ± {std_src:.1f} | budget {self.budget}")

        # ----- collect buffer & update distribution -----------------
        dyn_np, succ_np = self.train_sub.get_buffer()
        dyn = torch.as_tensor(dyn_np, dtype=self.current.params.dtype)
        succ = torch.as_tensor(succ_np, dtype=self.current.params.dtype)

        perf_val, kl_val = self._update_distribution(dyn, succ)
        ent_val = self.current.entropy().item()

        # ----- evaluate on TARGET domain ----------------------------
        target_env = CustomHopperDoraemon(
                dr_distribution=self.current,
                train_mode=False,
                return_threshold=self.return_threshold,
                domain="target")
        target_env.seed(GLOBAL_SEED)
        target_env = Monitor(target_env)

        mean_tgt, std_tgt = evaluate_policy(
            self.train_sub.model,
            target_env,
            n_eval_episodes=self.train_sub.n_eval_episodes,
            deterministic=True,
        )

        if self.verbose:
            print(
                f"Iter {self.iter}: entropy {ent_val:.8f} | perf {perf_val:.3f} | "
                f"KL {kl_val:.8f} | target R {mean_tgt:.2f} ± {std_tgt:.2f}"
            )

        self.iter += 1
        return True

    # --------------------------------------------------------------
    def run(self):
        while self.step():
            pass
        print(f"DORAEMON finished. Final entropy: {self.current.entropy().item():.3f}")


