"""
Implementation of DORAEMON: Domain Randomization via Entropy Maximization.
Supports both PPO algorithm via Stable-Baselines3.

"""
import os
import numpy as np
import torch
from torch.distributions.beta import Beta
from torch import digamma, lgamma
from scipy.special import polygamma 
from scipy.optimize import minimize, NonlinearConstraint
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from scipy.stats import beta
import pandas as pd
from typing import List
from IPython.display import display




import random

GLOBAL_SEED = 0
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

    def entropy_beta(self,a, b, lgamma, digamma):
        """Entropy of Beta(a,b)."""
        return (
            lgamma(a) + lgamma(b) - lgamma(a + b)
            - (a - 1) * digamma(a)
            - (b - 1) * digamma(b)
            + (a + b - 2) * digamma(a + b)
        )
    
    def entropy(self) -> float:
        """Analytic entropy of the multivariate Beta distribution on [m,M]."""
        a = self.params[0::2]; b = self.params[1::2]
        h = self.entropy_beta(a, b, lgamma, digamma).sum()
        # plus log(M−m)
        extra = sum(torch.log(torch.tensor(d["M"] - d["m"])) for d in self.distr)
        return float(h + extra)


    def kl_divergence(self, other: "DomainRandDistribution") -> float:
        """
        Analytic KL divergence D( P || Q ) between two multivariate Beta's
        P ~ Beta(a0,b0), Q ~ Beta(a1,b1) on each dimension.
        """
        a0 = self.params[0::2]
        b0 = self.params[1::2]
        a1 = other.params[0::2]
        b1 = other.params[1::2]

        # calculate KL dimension-wise:
        # D = ln B(a1,b1) - ln B(a0,b0)
        #   + (a0 - a1) ψ(a0) + (b0 - b1) ψ(b0)
        #   - (a0+b0 - a1 - b1) ψ(a0+b0)
        term = (
            lgamma(a1) + lgamma(b1) - lgamma(a1 + b1)
            - (lgamma(a0) + lgamma(b0) - lgamma(a0 + b0))
            + (a0 - a1) * digamma(a0)
            + (b0 - b1) * digamma(b0)
            - (a0 + b0 - a1 - b1) * digamma(a0 + b0)
        )
        return float(term.sum())
    
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
    
    # ------------------- for numpy compatibility ------------------------

    def _uni_pdf_np(self, x: np.ndarray, i: int, log: bool = False) -> np.ndarray:
        """numpy univariate pdf (or log pdf)."""
        params = self.as_numpy()              
        a, b = params[2*i], params[2*i+1]
        m, M = self.distr[i]["m"], self.distr[i]["M"]
        z = (x - m) / (M - m)
        if log:
            return beta.logpdf(z, a, b) - np.log(M - m)
        return beta.pdf(z, a, b) / (M - m)

    def pdf_np(self, x: np.ndarray, *, log: bool = False) -> np.ndarray:
        """NumPy multivariate pdf."""
        if x.ndim == 1:
            x = x[None, :]
        if log:
            out = np.zeros(x.shape[0])
            for i in range(self.ndims):
                out += self._uni_pdf_np(x[:, i], i, log=True)
            return out
        else:
            out = np.ones(x.shape[0])
            for i in range(self.ndims):
                out *= self._uni_pdf_np(x[:, i], i, log=False)
            return out



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
        lr=3e-4,
        n_eval_episodes=50,
        return_threshold=500.0,
        device="cpu",
        run_path="models",
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
            n_steps=2048,          
            batch_size=64,
            n_epochs=15,
            verbose=0,
            seed=GLOBAL_SEED,
            device=device,       
        )      

        
    # -------------------------------- train & evaluate on source
    def train(self, iter_idx: int, max_steps: int, model_name: any) -> tuple:
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
        self.model.save(os.path.join(self.run_path, f"{model_name}_iter{iter_idx+1}"))
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
        budget: int = 1_500_000,
        max_training_steps: int = 150_000,
        verbose: int = 1,
        model_name = 'model'
    ):
        self.train_sub = TrainingSubRtn(
            init_distr,
            return_threshold=return_threshold,
        )
        self.current = init_distr
        self.perf_lb = performance_lower_bound
        self.kl_ub = kl_upper_bound
        self.return_threshold = return_threshold
        self.budget = budget
        self.max_ts = max_training_steps
        self.verbose = verbose
        self.model_name = model_name
        self.iter = 0
        self.param_history = []
        init_ab = self.current.params.detach().cpu().numpy()
        self.param_history.append(init_ab.tolist())
        self.entropy_history = []
        self.kl_history = [0]
        self.perf_history = [np.nan] 
        self.target_mean = []
        self.target_std = []
        self.source_mean = []
        self.source_std = []
        self.target_env = CustomHopperDoraemon(
                train_mode=False,
                return_threshold=self.return_threshold,
                domain="target")
        self.target_env.seed(GLOBAL_SEED)
        self.target_env = Monitor(self.target_env)

        initial_entropy = self.current.entropy()
        self.entropy_history.append(initial_entropy)
        if self.verbose:
            print(f"Initial entropy: {initial_entropy:.3f}")

    # --------------------------------------------------------------
    def _update_distribution(self, dyn: torch.Tensor, succ: torch.Tensor):
        """Solve the constrained optimisation to update Beta params."""

        bounds = self.current.get_stacked_bounds()
        dyn_np = dyn.detach().numpy()
        succ_np = succ.detach().numpy()

        
        def _build_candidate(x_opt_np: np.ndarray) -> DomainRandDistribution:
            """Build a candidate distribution from the optimised parameters."""
            beta_par = DomainRandDistribution.sigmoid_param(x_opt_np, MIN_PARAM, MAX_PARAM)
            return DomainRandDistribution.beta_from_stacked(bounds, beta_par)

        # ----- performance constraint (importance sampling) ----------
        def perf_fn(x_opt):
            cand = _build_candidate(x_opt)
            w_log = cand.pdf_np(dyn_np, log=True) - self.current.pdf_np(dyn_np, log=True)
            w = np.exp(w_log)
            w_norm = w/np.sum(w)
            return float(np.dot(w_norm, succ_np))

        perf_cons = NonlinearConstraint(perf_fn, lb=self.perf_lb, ub=np.inf)

        # ----- KL constraint ----------------------------------------
        def kl_fn(x_opt):
            cand = _build_candidate(x_opt)
            return self.current.kl_divergence(cand)

        kl_cons = NonlinearConstraint(kl_fn, lb=-np.inf, ub=self.kl_ub)

        # ----- objective: maximise entropy (= minimise -entropy) -----
        def objective(x_opt):
            cand = _build_candidate(x_opt)
            return -cand.entropy()
        

        # ----- gradients for scipy optimization -----
        def grad_entropy_obj(x, MIN_PARAM, MAX_PARAM):
            """
            Compute ∇ₓ[–entropy(Beta(sigmoid(x)))].

            """
            # 1) Map x -> [MIN_PARAM, MAX_PARAM] via sigmoid
            sig = DomainRandDistribution.sigmoid_param(x, MIN_PARAM, MAX_PARAM)
            a = sig[0::2]  # a parameters
            b = sig[1::2]  # b parameters


            # 2) Derivative of the sigmoid mapping wrt x
            inv = np.exp(-x)
            dsig = (MAX_PARAM - MIN_PARAM) * inv / (1 + inv)**2
            da = dsig[0::2]
            db = dsig[1::2]

            # 3) Compute trigamma values (ψ₁) at a, b, and a+b
            psi1_a  = polygamma(1, a)
            psi1_b  = polygamma(1, b)
            psi1_ab = polygamma(1, a + b)

            # 4) Partial derivatives of the Beta entropy H(a,b)
            #    ∂H/∂a = (a+b−2)·ψ₁(a+b) − (a−1)·ψ₁(a)
            #    ∂H/∂b = (a+b−2)·ψ₁(a+b) − (b−1)·ψ₁(b)
            dH_da = (a + b - 2) * psi1_ab - (a - 1) * psi1_a
            dH_db = (a + b - 2) * psi1_ab - (b - 1) * psi1_b

            # 5) Chain rule: ∇ₓ[−H] = −(∂H/∂a)*da  for x[0::2]
            #                    −(∂H/∂b)*db  for x[1::2]
            grad = np.zeros_like(x)
            grad[0::2] = -dH_da * da
            grad[1::2] = -dH_db * db

            return grad
        
        def grad_perf_obj(x, dyn_np, succ_np, bounds, curr: DomainRandDistribution):
            """
            ∇ₓ [-perf_fn(x)] where
            perf_fn(x) = E_curr [ exp(w_log(x)) * succ_np ],
            w_log(x) = log p_cand(dyn|x) − log p_curr(dyn)
            """
            ab   = DomainRandDistribution.sigmoid_param(x, MIN_PARAM, MAX_PARAM)
            cand = DomainRandDistribution.beta_from_stacked(bounds, ab)

            # log‐pdfs and raw weights
            logp_cand = cand.pdf_np(dyn_np, log=True)
            logp_curr = curr.pdf_np(dyn_np, log=True)
            w_log     = logp_cand - logp_curr
            w         = np.exp(w_log)              

            # SNIS estimate
            sum_w     = w.sum()
            perf_hat  = float((w * succ_np).sum() / sum_w)

            # data normalisation
            # dyn_np: shape (K, ndims)
            z = (dyn_np - np.array([d['m'] for d in curr.distr])) / \
            np.array([d['M']-d['m'] for d in curr.distr])  # shape (K,ndims)

            # precompute digamma for log‐pdf derivative
            a = ab[0::2]; b = ab[1::2]
            psi_a  = polygamma(0, a)
            psi_b  = polygamma(0, b)
            psi_ab = polygamma(0, a+b)

            # derivative of sigmoid
            inv = np.exp(-x)
            dsig = (MAX_PARAM - MIN_PARAM) * inv / (1 + inv)**2
            da   = dsig[0::2]
            db   = dsig[1::2]

            # allocate gradient
            grad = np.zeros_like(x)

            # loop dims
            K, nd = dyn_np.shape
            for i in range(curr.ndims):
                # ∂ log p_beta / ∂ a, ∂ b for each sample k
                lnz   = np.log(z[:, i].clip(1e-12, 1-1e-12))
                ln1z  = np.log((1-z[:, i]).clip(1e-12, 1-1e-12))
                dlogp_da = lnz   - psi_a[i]  + psi_ab[i]   # shape (K,)
                dlogp_db = ln1z  - psi_b[i]  + psi_ab[i]

                # combine into ∂ log p_cand / ∂ x
                # note: x[2i]→a_i, x[2i+1]→b_i
                # ∂ logp/∂x_j = dlogp/∂a * da  or * db
                term_a = dlogp_da * da[i]  # shape (K,)
                term_b = dlogp_db * db[i]

                # weights for SNIS gradient: w_k*(s_k - perf_hat)/sum_w
                coeff = w * (succ_np - perf_hat) / sum_w  # shape (K,)

                # now sum_k coeff[k] * term_a[k] etc.
                grad[2*i]   = np.dot(coeff, term_a)
                grad[2*i+1] = np.dot(coeff, term_b)

            return -grad  # shape (2*ndims,)

        # ----- optimisation start point -----------------------------
        x0 = self.current.as_numpy()
        x0 = DomainRandDistribution.inverse_sigmoid_param(x0, MIN_PARAM, MAX_PARAM)

        # FIRST CHECK: performance constraint
        perf0 = perf_fn(x0)

        if perf0 < self.perf_lb:
            # if perf< perf_lb go directly to backup
            print("Not feasible initial distribution. Starting backup optimization.")
            res = minimize(
                fun=lambda x: -perf_fn(x),
                x0=x0,
                jac= lambda x: grad_perf_obj(x, dyn_np, succ_np, bounds, self.current),
                method="trust-constr",
                constraints=[kl_cons],
                options={"maxiter": 100}
            )
            perf_new = perf_fn(res.x)
            kl_new = kl_fn(res.x)
        else:
            # otherwise, maximise entropy with both constraints
            print("Initial distribution is feasible. Starting entropy maximization.")
            res = minimize(
                fun=objective,
                x0=x0,
                method="trust-constr",
                jac=lambda x: grad_entropy_obj(x, MIN_PARAM, MAX_PARAM),
                constraints=[perf_cons, kl_cons],
                options={"maxiter": 100}
            )
            perf_new = perf_fn(res.x)
            kl_new = kl_fn(res.x)
            # if fail, go to backup
            if perf_new < self.perf_lb or kl_new > self.kl_ub:
                print("Entropy maximization failed. Starting backup optimization.")
                res = minimize(
                    fun=lambda x: -perf_fn(x),
                    x0=x0,
                    jac= lambda x: grad_perf_obj(x, dyn_np, succ_np, bounds, self.current),
                    method="trust-constr",
                    constraints=[kl_cons],
                    options={"maxiter": 100}
                )
                perf_new = perf_fn(res.x)
                kl_new = kl_fn(res.x)

        # update distribution
        new_ab = DomainRandDistribution.sigmoid_param(res.x, MIN_PARAM, MAX_PARAM)
        new_t  = torch.as_tensor(new_ab, dtype=self.current.params.dtype, device=self.current.params.device)
        with torch.no_grad():
            self.current.params.data.copy_(new_t)
            self.current._build()

        # record params 
        self.param_history.append(new_ab.tolist())


        return perf_new, kl_new

    # --------------------------------------------------------------
    def step(self) -> bool:
        """One DORAEMON iteration: train → optimise → log → eval on target."""
        if self.budget <= 0:
            return False

        # ----- RL training on source --------------------------------
        max_steps = min(self.budget, self.max_ts)
        mean_src, std_src, used_ts = self.train_sub.train(self.iter, max_steps, self.model_name)
        self.budget -= used_ts
        self.source_mean.append(mean_src)
        self.source_std.append(std_src)

        if self.verbose:
            print(f"Iter {self.iter+1}: source R {mean_src:.1f} ± {std_src:.1f} | budget {self.budget}")

        # ----- evaluate on TARGET domain ----------------------------

        mean_tgt, std_tgt = evaluate_policy(
            self.train_sub.model,
            self.target_env,
            n_eval_episodes=self.train_sub.n_eval_episodes,
            deterministic=True,
        )

        if self.verbose:
            print(
                f" target R {mean_tgt:.2f} ± {std_tgt:.2f}"
            ) 


        # ----- collect buffer & update distribution -----------------
        dyn_np, succ_np = self.train_sub.get_buffer()
        dyn = torch.as_tensor(dyn_np, dtype=self.current.params.dtype)
        succ = torch.as_tensor(succ_np, dtype=self.current.params.dtype)

        perf_val, kl_val = self._update_distribution(dyn, succ)
        ent_val = self.current.entropy()


        if self.verbose:
            print(
                f"Iter {self.iter+1}: new entropy {ent_val:.8f} | new perf {perf_val:.3f} | "
            ) 
        
        self.entropy_history.append(ent_val)
        self.kl_history.append(kl_val)
        self.target_mean.append(mean_tgt)
        self.target_std.append(std_tgt)
        self.perf_history.append(perf_val)

        self.iter += 1
        return True

    # --------------------------------------------------------------
    def run(self):
        while self.step():
            pass

        final_source_mean, final_source_std, _ = self.train_sub.train(self.iter, self.max_ts, self.model_name)

        if self.verbose:
            print(f"Iter {self.iter+1}: source R {final_source_mean:.1f} ± {final_source_std:.1f} | budget {self.budget}")

        self.source_mean.append(final_source_mean)
        self.source_std.append(final_source_std)


        final_target_mean, final_target_std = evaluate_policy(
                self.train_sub.model,
                self.target_env,
                n_eval_episodes=self.train_sub.n_eval_episodes,
                deterministic=True,
        )

        if self.verbose:
            print(
                f" target R {final_target_mean:.2f} ± {final_target_std:.2f}"
            ) 

        self.target_mean.append(final_target_mean)
        self.target_std.append(final_target_std)    

        
        
        print(f"DORAEMON finished. Final entropy: {self.current.entropy():.3f}")
        print(f"Parameters history: {self.param_history} ")

        iters = [f"Iter {i}" for i in range(len(self.entropy_history))]
        df = pd.DataFrame({
            'Entropy'     : self.entropy_history,
            'KL vs prev'  : self.kl_history,
            'Perf Est'    : self.perf_history,
            'Source Mean' : self.source_mean,
            'Source Std'  : self.source_std,
            'Target Mean' : self.target_mean,
            'Target Std'  : self.target_std,
        }, index=iters)

        display(df)






