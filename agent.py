import numpy as np
import pandas as pd
from config import Config
import math


class QLearning:
    def __init__(self, n_states, n_actions, cfg: Config):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = cfg.lr  # 学习率
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.actions = range(self.n_actions)
        self.q_table = pd.DataFrame(np.zeros((self.n_states, self.n_actions)))

    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon - self.epsilon_end) * \
                       math.exp(-1.0 * self.sample_count / self.epsilon_decay)

        # e-greedy algorithm
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.iloc[state, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def update(self, state, action, reward, next_state, terminated, truncated):
        q_predict = self.q_table.iloc[state, action]

        if not terminated:
            q_target = reward + self.gamma * \
                       self.q_table.iloc[next_state, :].max()
        else:
            q_target = reward

        self.q_table.iloc[state, action] += self.gamma * \
                                            (q_target - q_predict)
