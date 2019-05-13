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
import math

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


def n_step_expected_value_delta(mdp, max_episode, alpha = 0.1, gamma = 0.9, epsilon = 0.1, n = 100, seed=0, schedule_ks=True, schedule_alpha=False):
    behaviour_policy = random_behaviour_policy
    behaviour_policy_probs = random_behaviour_policy_probs
    # pbar = tqdm(total=max_episode)
    mdp.seed(seed)

    # gamma_0=0 then gamma_{z+1} = (gamma_z +1) /2

    alpha_gamma_ks = []
    current_gamma = 0.0
    while current_gamma < gamma:
        if schedule_ks:
            next_k = math.ceil(1./(1.-current_gamma)) #(need plus one otherwise we add bias/error)
        else:
            next_k = n
        if schedule_alpha:
            effective_horizon = float(1./(1.-current_gamma))
            rescale_factor  = max(float(n) / effective_horizon, 1.0)
            lr = min(max(alpha * rescale_factor, alpha), 1.0)
        else:
            lr = alpha
        alpha_gamma_ks.append((lr, current_gamma, min(next_k, n)))
        current_gamma = ((current_gamma + 1.)/2.)

    # Q is now broken up into the deltas
    alpha_gamma_ks.append((alpha, gamma, n))

    V = np.array([[0 for j in range(mdp.observation_space.n)] for z in range(len(alpha_gamma_ks))] , dtype=float)

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
        next_state = mdp.reset()
        stored_actions[0] = behaviour_policy(V, next_state, mdp.action_space.n)
        reward_for_episode = 0

        while tau < (T-1):
            t += 1
            if t < T:
                # take action A_t
                cur_state = next_state
                # Observe and store the next reward R_{t+1} and next state S_{t+1}
                next_state, reward, done, info = mdp.step(stored_actions[t % n])

                stored_rewards[(t) % n] = reward
                stored_states[(t) % n] = cur_state


                total_reward += reward
                reward_for_episode += reward

                if done:
                    T = t + 1
                else:
                    stored_actions[(t+1) % n] = behaviour_policy(V, next_state, mdp.action_space.n)

            tau = t - n + 1
            if tau >= 0 and not done:
                # TODO: sum passed to policies instead of raw Q list
            
                G = []
                for j in range(len(alpha_gamma_ks)):
                    alpha, gamma, k_step = alpha_gamma_ks[j]
                    #TODO: there may be an off by one in the trace
                    if j==0:
                        # normal
                        trace = np.sum([gamma**(i-tau) * (stored_rewards[i%n]) for i in range(tau, min(tau+k_step, T))])
                    else:
                        # differential
                        lower_gamma = alpha_gamma_ks[j-1][1]
                        # TODO: is there off by one here?
                        trace = np.sum([(gamma**(i-tau) - lower_gamma**(i-tau)) * (stored_rewards[i%n]) for i in range(tau+1, min(tau+k_step, T))])
                    G.append(trace)

                    
                # this should have the other estimator sum at the end
                for j in range(len(alpha_gamma_ks)):
                    alpha, gamma, k_step = alpha_gamma_ks[j]
                    if tau + k_step < T:
                        bootstrap_state = stored_states[(tau + k_step) % n] if k_step < n else next_state
                        if j == 0:
                            difference = ( gamma ** k_step )  * V[j][bootstrap_state]
                        else:
                            lower_gamma = alpha_gamma_ks[j-1][1]
                            # TODO: do we need importance sampling per differential function
                            difference = ( gamma ** k_step ) * V[j][bootstrap_state] \
                                + ( gamma ** k_step - lower_gamma**k_step) * np.sum(V[:j], axis=0)[bootstrap_state]
                        G[j] = G[j] +  difference

                s_tau = stored_states[tau % n]
                a_tau = stored_actions[tau %  n]

                for l in range(len(G)):
                    V[l][s_tau] += alpha_gamma_ks[l][0] * (G[l]- V[l][s_tau])
                Vs.append(copy.deepcopy(np.sum(V, axis=0)))

        if reward_for_episode > max_reward:
            max_reward = reward_for_episode

        rewards_per_episode.append(reward_for_episode)
        # TODO: the variance doesn't make sense in this context
        V_variances.append(np.var(np.sum(V, axis=0)))
        
        n_episode += 1
        # pbar.update(1)

    # pbar.close()
    return Vs, total_reward/max_episode, max_reward, rewards_per_episode, V_variances
