import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import scipy.signal
import pickle

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.pi_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _distribution(self, obs):
        mu = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.v = Critic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            pi = self.pi._distribution(obs_tensor)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs_tensor)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            pi = self.pi._distribution(obs_tensor)
            a = pi.sample()
        return a.numpy()

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]

    def get(self):
        # RESTORED: Assertion to ensure buffer is full with fixed-step epochs
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

class PPOAgent:
    def __init__(self, observation_space, action_space, ac_kwargs=None, seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01):
        if ac_kwargs is None:
            ac_kwargs = {}
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.actor_critic = MLPActorCritic(observation_space, action_space, **ac_kwargs)
        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.actor_critic.v.parameters(), lr=vf_lr)
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.buf = PPOBuffer(observation_space.shape, action_space.shape, self.steps_per_epoch, gamma, lam)

    def get_action(self, obs):
        return self.actor_critic.step(obs)

    def compute_losses(self, data):
        obs, act, adv, ret, logp_old = data['obs'], data['act'], data['adv'], data['ret'], data['logp']
        pi, logp = self.actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        loss_v = ((self.actor_critic.v(obs) - ret) ** 2).mean()
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        return loss_pi, loss_v, approx_kl, ent, clipfrac

    def update_learning_rates(self, new_pi_lr, new_vf_lr):
        for param_group in self.pi_optimizer.param_groups:
            param_group['lr'] = new_pi_lr
        for param_group in self.vf_optimizer.param_groups:
            param_group['lr'] = new_vf_lr

    def update(self):
        data = self.buf.get()
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, _, approx_kl, ent, clipfrac = self.compute_losses(data)
            if approx_kl > 1.5 * self.target_kl:
                print(f'Early stopping policy update at step {i} due to reaching max KL divergence.')
                break
            loss_pi.backward()
            self.pi_optimizer.step()
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            _, loss_v, _, _, _ = self.compute_losses(data)
            loss_v.backward()
            self.vf_optimizer.step()
        final_loss_pi, final_loss_v, final_approx_kl, final_ent, final_clipfrac = self.compute_losses(data)
        return {'loss_pi': final_loss_pi.item(), 'loss_v': final_loss_v.item(), 'approx_kl': final_approx_kl, 'ent': final_ent, 'clipfrac': final_clipfrac}

    def save_weights(self, filename="ppo_drone_agent.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.actor_critic.state_dict(), f)
        print(f"Agent weights saved to {filename}")

    def load_weights(self, filename="ppo_drone_agent.pkl"):
        try:
            with open(filename, 'rb') as f:
                state_dict = pickle.load(f)
            self.actor_critic.load_state_dict(state_dict)
            print(f"Agent weights loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Weights file '{filename}' not found. Ensure training was run and weights were saved.")
        except Exception as e:
            print(f"Error loading weights: {e}")
