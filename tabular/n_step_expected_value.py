# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import sys
from tqdm import tqdm
import copy
# In this scenario we're always doing random walks

def random_behaviour_policy(Q, s, nA, epsilon=.3, random_walk=False):
    """
    Recall that off-policy learning is learning the value function for
    one policy, \pi, while following another policy, \mu. Often, \pi is
    the greedy policy for the current action-value-function estimate,
    and \mu is a more exploratory policy, perhaps \epsilon-greedy.
    In order to use the data from \pi we must take into account the
    difference between the two policies, using their relative
    probability of taking the actions that were taken.
    NOTE: Some parts were modified from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA.ipynb
    """
    return np.random.choice(range(nA))

def random_behaviour_policy_probs(Q, s, nA, epsilon=.3):
    return np.ones(nA, dtype=float) / nA


def n_step_expected_value(mdp, max_episode, alpha = 0.1, gamma = 0.9, epsilon = 0.1, n = 100, seed=0):
    behaviour_policy = random_behaviour_policy
    behaviour_policy_probs = random_behaviour_policy_probs
    mdp.seed(seed)
    pbar = tqdm(total=max_episode)
    V = np.array([0 for j in range(mdp.observation_space.n)], dtype=float)
    print("VSHAPE")
    print(V.shape)
    # gamma_0=0 then gamma_{z+1} = (gamma_z +1) /2

    n_episode = 0
    rewards_per_episode = []
    V_variances = []
    max_reward = 0
    total_reward = 0
    Vs = []

    while n_episode < max_episode:

        # initializations
        T = sys.maxsize
        tau = 0
        t = -1
        stored_actions = {}
        stored_rewards = {}
        stored_states = {}

        # With prob epsilon, pick a random action

        stored_states[0] = mdp.reset()
        stored_actions[0] = behaviour_policy(V, stored_states[0], mdp.action_space.n)
        reward_for_episode = 0

        while tau < (T-1):
            t += 1
            if t < T:
                # take action A_t

                # Observe and store the next reward R_{t+1} and next state S_{t+1}
                next_state, reward, done, info = mdp.step(stored_actions[t % n])

                stored_rewards[(t+1) % n] = reward
                stored_states[(t+1) % n] = next_state


                total_reward += reward
                reward_for_episode += reward

                if done:
                    T = t + 1
                else:
                    stored_actions[(t+1) % n] = behaviour_policy(V, next_state, mdp.action_space.n)

            tau = t - n + 1
            if tau >= 0:
                trace = np.sum([gamma**(i-tau-1) * (stored_rewards[i%n]) for i in range(tau+1, min(tau+n-1, T-1)+1)])
                    
                # this should have the other estimator sum at the end
                if tau + n < T:
                    difference = ( gamma ** n )  * V[stored_states[(tau+n) % n]]
                else:
                    difference = 0.0

                s_tau = stored_states[tau % n]
                a_tau = stored_actions[tau %  n]

                V[s_tau] += alpha * ((trace + difference) - V[s_tau])

        if reward_for_episode > max_reward:
            max_reward = reward_for_episode

        rewards_per_episode.append(reward_for_episode)
        V_variances.append(np.var(V))
        Vs.append(copy.deepcopy(V))

        n_episode += 1
        pbar.update(1)

    pbar.close()
    return Vs, total_reward/max_episode, max_reward, rewards_per_episode, V_variances
