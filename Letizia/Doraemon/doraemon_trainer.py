import numpy as np
from typing import Callable, List, Tuple
from scipy.special import beta as B, digamma

class DoraemonTrainer:
    def __init__(
        self,
        policy,
        env_factory: Callable[[np.ndarray], object],
        phi_init: np.ndarray,
        alpha_success: float = 0.8,
        kl_eps: float = 0.1,
        lr_phi: float = 1e-3,
        batch_size: int = 64,
        max_iter: int = 1000,
        algorithm: str = 'reinforce',
    ):
        # Store policy, factory, and hyperparameters
        self.policy = policy
        self.env_factory = env_factory
        self.phi = phi_init  # shape (d,2) for Beta(a,b) parameters
        self.alpha = alpha_success
        self.kl_eps = kl_eps
        self.lr_phi = lr_phi
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.algorithm = algorithm
        self.agent = None

    def sample_xis(self) -> np.ndarray:
        """
        Sample a batch of parameter vectors xi from the current Beta distribution nu_phi.
        Returns:
            xis: array of shape (batch_size, d)
        """
        a = self.phi[:, 0]
        b = self.phi[:, 1]
        # Sample each dimension independently
        xis = np.stack([np.random.beta(a[i], b[i], size=self.batch_size)
                         for i in range(len(a))], axis=1)
        return xis

    def collect_rollouts(self, xis: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
        """
        For each xi, create an environment and run a full episode, storing outcomes.
        Returns a list of (xi, success_flag).
        """
        from agent import Agent
        # Initialize agent on first use
        if self.agent is None:
            self.agent = Agent(self.policy)
        data = []
        for xi in xis:
            env = self.env_factory(xi)
            state = env.reset()
            self.agent.reset_I()
            done = False
            info = {}
            # Clear stored trajectories in agent
            # Run episode
            while not done:
                action, logp = self.agent.get_action(state)
                next_state, reward, done, info = env.step(action.detach().cpu().numpy())
                self.agent.store_outcome(state, next_state, logp, reward, done)
                state = next_state
            # Record whether this episode succeeded
            success_flag = info.get("success", False)
            data.append((xi, success_flag))
        return data

    def policy_update(self, data):
        """
        Update the policy using collected trajectories via the specified algorithm.
        """
        self.agent.update_policy(self.algorithm)

    def estimate_success(self, xis: np.ndarray, successes: np.ndarray, phi_candidate: np.ndarray) -> float:
        """
        Estimate the success probability G for a candidate phi via importance sampling.
        """
        a_old, b_old = self.phi[:,0], self.phi[:,1]
        a_new, b_new = phi_candidate[:,0], phi_candidate[:,1]
        # Compute probability densities under old and new Betas
        old_pdf = np.prod([xis[:,i]**(a_old[i]-1)*(1-xis[:,i])**(b_old[i]-1)/B(a_old[i],b_old[i])
                            for i in range(len(a_old))], axis=0)
        new_pdf = np.prod([xis[:,i]**(a_new[i]-1)*(1-xis[:,i])**(b_new[i]-1)/B(a_new[i],b_new[i])
                            for i in range(len(a_new))], axis=0)
        weights = new_pdf / (old_pdf + 1e-12)
        return float(np.mean(weights * successes))

    def phi_update(self, data):
        """
        Perform a constrained update on phi to maximize entropy while
        maintaining a minimum success rate and limiting KL divergence.

        Steps:
        1. Compute empirical success rate hatG under current phi.
        2. Compute gradient of entropy for each Beta(a,b).
        3. Propose phi_candidate via gradient ascent on entropy.
        4. Estimate hatG_candidate via importance sampling.
        5. If hatG_candidate >= alpha, accept candidate with small clipping to enforce KL.
        6. Otherwise, backtrack along the update direction until success >= alpha.
        """
        # Unpack data
        xis = np.stack([d[0] for d in data])
        successes = np.array([d[1] for d in data], dtype=float)
        # Empirical success under old phi
        hatG = float(np.mean(successes))
        # Entropy gradient for Beta: dH/da and dH/db
        a, b = self.phi[:,0], self.phi[:,1]
        psi_ab = digamma(a+b)
        grad_a = psi_ab - digamma(a) + (a-1)/(a+b) - (b-1)/(a+b)
        grad_b = psi_ab - digamma(b) + (b-1)/(a+b) - (a-1)/(a+b)
        entropy_grad = np.stack([grad_a, grad_b], axis=1)
        # Propose candidate update
        phi_candidate = np.clip(self.phi + self.lr_phi * entropy_grad, 1e-3, None)
        # Estimate success for candidate
        hatG_candidate = self.estimate_success(xis, successes, phi_candidate)
        # Constrained accept or backtrack
        if hatG_candidate >= self.alpha:
            # Clip parameter changes to approximate KL trust-region
            delta = phi_candidate - self.phi
            max_change = self.kl_eps
            delta = np.clip(delta, -max_change, max_change)
            self.phi += delta
        else:
            # Backtracking line search
            step = 1.0
            direction = phi_candidate - self.phi
            for _ in range(10):
                phi_try = np.clip(self.phi + step * direction, 1e-3, None)
                if self.estimate_success(xis, successes, phi_try) >= self.alpha:
                    self.phi = phi_try
                    break
                step *= 0.5

    def train(self):
        """
        Main training loop:
        repeat: sample_xis -> collect_rollouts -> policy_update -> phi_update
        """
        for iteration in range(self.max_iter):
            xis = self.sample_xis()
            data = self.collect_rollouts(xis)
            self.policy_update(data)
            self.phi_update(data)
