"""Importable fake env for vec tests. Must be a real module (not defined inside
a test file) so that 'spawn' subprocess workers can re-import it to unpickle."""
import numpy as np


class FakeEnv:
    def __init__(self, seed=0, n_agents=3, n_actions=5, obs_dim=8, state_dim=10):
        self.seed = seed
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.rng = np.random.RandomState(seed)
        self.ep_len = 3 + (seed % 5)
        self.reset()

    def get_env_info(self):
        return {"n_agents": self.n_agents, "n_actions": self.n_actions,
                "obs_shape": self.obs_dim, "state_shape": self.state_dim,
                "episode_limit": 100}

    def get_obs_move_feats_size(self):
        return 2

    def get_obs_enemy_feats_size(self):
        return (2, 2)

    def reset(self):
        self.t = 0
        return self.get_obs(), self.get_state()

    def get_obs(self):
        return np.full((self.n_agents, self.obs_dim), float(self.seed),
                       dtype=np.float32)

    def get_state(self):
        return np.full(self.state_dim, float(self.seed), dtype=np.float32)

    def get_avail_actions(self):
        av = np.ones((self.n_agents, self.n_actions), dtype=np.float32)
        av[:, 0] = 0.0
        return av

    def step(self, actions):
        self.t += 1
        terminated = self.t >= self.ep_len
        return 1.0, terminated, {"battle_won": terminated and (self.seed % 2 == 0),
                                 "seed": self.seed}

    def close(self):
        pass


def factory(rank):
    return lambda: FakeEnv(seed=100 + rank)
