# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
from gym import spaces
from gym.utils import seeding

class NRingEnv(gym.Env):
    """n-Ring environment

    This is based on the ring MDP environment found in:

    Kearns, Michael J., and Satinder P. Singh. "Bias-Variance Error Bounds for Temporal Difference Updates." In COLT, pp. 142-147. 2000.
    

    ``In this problem, we have a Markov process with 5 states arranged in a ring. At each step,
    there is probability 0.05 that we remain in our current state, and probability 0.95 that we
    advance one state clockwise around the ring. (Note that since we are only concerned with the
    evaluation of a fixed policy, we have simply defined a Markov process rather than a Markov
    decision process.) Two adjacent states on the ring have reward +1 and -1 respectively,
    while the remaining states have reward 0. The standard random walk problem has a chain
    of states, with an absorbing state at each end; here we chose a ring structure simply to avoid
    asymmetries in the states induced by the absorbing state.''
    """
    def __init__(self, n=5, slip=0.05):
        assert n>=5
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.state = 0  # Start at beginning of the chain
        self.rewarding_states = int(n/2), int(n/2) + 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.count = 0
        self.seed()
        self.max_count = 500

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_reward(self, state):
        if state == self.rewarding_states[0]:
            return 1
        elif state == self.rewarding_states[1]:
            return -1
        else:
            return 0

    def step(self, action):
        assert self.action_space.contains(action)
        self.count += 1
        # We don't actually care about the action.
        if self.np_random.rand() < self.slip:
            return self.state, self._get_reward(self.state), self.count >= self.max_count, {} 
        else:
            self.state = (self.state + 1) % self.n
            return self.state, self._get_reward(self.state), self.count >= self.max_count, {}

    def reset(self):
        self.state = 0
        self.count = 0
        return self.state