# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from n_step_expected_value_delta import n_step_expected_value_delta
from n_step_expected_value import n_step_expected_value

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gym
from ring_env import NRingEnv
from get_ring_env_value import PRECOMPUTED_VALS
import matplotlib.style as style
# style.use('fivethirtyeight')
from scipy import stats
# simplearg.py
import argparse
import pickle
import os
import multiprocessing
import sys


def average_error(gamma, seed, vals):
    real_vals = PRECOMPUTED_VALS[gamma]
    assert len(PRECOMPUTED_VALS[gamma]) == 5
    value_errors = []
    for val in vals:
        errors = []
        for state, value in enumerate(val):
            errors.append(np.abs(real_vals[state] - value))
        value_errors.append(np.mean(errors))
    return value_errors

gammas = [.75, .875, .9375, .96875, .984375, .992, .996]
ks = [4, 8, 16, 32, 64, 128, 256]
RUNS = []
for g in gammas:
    for k in ks:
        RUNS.append((g, k))

def main(args):

    gw = NRingEnv()

    # for gamma in [.75, .875, .9375, .96875, .984375, .992, .996]:
    # python different_alpha_value_only_experiment.py --gamma .75 --k 4 &> log.75.4.log & disown
    # python different_alpha_value_only_experiment.py --gamma .75 --k 8 &> log.75.8.log & disown
    # python different_alpha_value_only_experiment.py --gamma .75 --k 16 &> log.75.16.log & disown
    # python different_alpha_value_only_experiment.py --gamma .75 --k 32 &> log.75.32.log & disown
    # python different_alpha_value_only_experiment.py --gamma .75 --k 64 &> log.75.64.log & disown
    # python different_alpha_value_only_experiment.py --gamma .75 --k 128 &> log.75.128.log & disown
    # python different_alpha_value_only_experiment.py --gamma .75 --k 256 &> log.75.256.log & disown
    # python different_alpha_value_only_experiment.py --gamma .875 --k 4 &> log.875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .875 --k 8 &> log.875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .875 --k 16 &> log.875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .875 --k 32 &> log.875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .875 --k 64 &> log.875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .875 --k 128 &> log.875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .875 --k 256 &> log.875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .9375 --k 4 &> log.9375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .9375 --k 8 &> log.9375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .9375 --k 16&> log.9375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .9375 --k 32 &> log.9375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .9375 --k 64 &> log.9375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .9375 --k 128 &> log.9375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .9375 --k 256 &> log.9375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .96875 --k 4&> log.96875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .96875  --k 8&> log.96875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .96875 --k 16 &> log.96875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .96875 --k 32 &> log.96875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .96875 --k 64 &> log.96875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .96875  --k 128 &> log.96875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .96875 --k 256 &> log.96875.log & disown
    # python different_alpha_value_only_experiment.py --gamma .984375 --k 4&> log.984375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .984375 --k 8&> log.984375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .984375 --k 16&> log.984375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .984375 --k 32&> log.984375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .984375 --k 64&> log.984375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .984375 --k 128&> log.984375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .984375 --k 256&> log.984375.log & disown
    # python different_alpha_value_only_experiment.py --gamma .992 --k 4 &> log.992.log & disown
    # python different_alpha_value_only_experiment.py --gamma .992 --k 8 &> log.992.log & disown
    # python different_alpha_value_only_experiment.py --gamma .992 --k 16 &> log.992.log & disown
    # python different_alpha_value_only_experiment.py --gamma .992 --k 32 &> log.992.log & disown
    # python different_alpha_value_only_experiment.py --gamma .992 --k 64 &> log.992.log & disown
    # python different_alpha_value_only_experiment.py --gamma .992 --k 128 &> log.992.log & disown
    # python different_alpha_value_only_experiment.py --gamma .992 --k 256&> log.992.log & disown
    # python different_alpha_value_only_experiment.py --gamma .996 --k 4&> log.996.log & disown
    # python different_alpha_value_only_experiment.py --gamma .996 --k 8 &> log.996.log & disown
    # python different_alpha_value_only_experiment.py --gamma .996 --k 16 &> log.996.log & disown
    # python different_alpha_value_only_experiment.py --gamma .996 --k 32 &> log.996.log & disown
    # python different_alpha_value_only_experiment.py --gamma .996 --k 64 &> log.996.log & disown
    # python different_alpha_value_only_experiment.py --gamma .996 --k 128&> log.996.log & disown
    # python different_alpha_value_only_experiment.py --gamma .996 --k 256&> log.996.log & disown
    if args.run_index is not None:
        gamma, k = RUNS[args.run_index]
    else:
        gamma= args.gamma
        k =args.k
    # for k in [4, 8,  16, 32, 64, 128, 256]:

    average_reward_n_step_expected_value_scheduled_delta = []
    all_rewards_per_episode_n_step_expected_value_scheduled_delta = []
    q_var_n_step_expected_value_scheduled_delta = []
    q_var_n_step_expected_value_scheduled_delta_errors = []


    average_reward_n_step_expected_value = []
    all_rewards_per_episode_n_step_expected_value = []
    q_var_n_step_expected_value = []
    q_var_n_step_expected_value_errors = []
    vals = []
    vals1 = []


    epsilon = .1

    n=3000
    alphas = [x for x in np.arange(0.0, 1., .1)]
    alphas[0] = .01
    # import pdb; pdb.set_trace()

    number_of_runs = 200

    with Parallel(n_jobs=2, backend = 'multiprocessing') as parallel:
        for r in range(number_of_runs):
            n_step_expected_value_results = parallel(delayed(n_step_expected_value)(gw, n, alpha, gamma, epsilon, n=k, seed=r) for alpha in alphas)
            for result in n_step_expected_value_results:
                vals.append(result[0])
                average_reward_n_step_expected_value.append(result[1])
                q_var_n_step_expected_value.append(result[4])
                all_rewards_per_episode_n_step_expected_value.append(result[3])
                q_var_n_step_expected_value_errors.append(average_error(gamma, r, result[0]))
            print("Done nstep expected sarsa")

            n_step_expected_value_scheduled_delta_results = parallel(delayed(n_step_expected_value_delta)(gw, n, alpha, gamma, epsilon, n=k, seed=r, schedule_ks=True) for alpha in alphas)
            # Q, average_reward, max_reward, all_rewards, Q_variances = n_step_expected_value_delta(gw, max_episode, epsilon=epsilon, alpha=alpha, n = n_step)
            for result in n_step_expected_value_scheduled_delta_results:
                vals1.append(result[0])
                average_reward_n_step_expected_value_scheduled_delta.append(result[1])
                q_var_n_step_expected_value_scheduled_delta.append(result[4])
                all_rewards_per_episode_n_step_expected_value_scheduled_delta.append(result[3])
                q_var_n_step_expected_value_scheduled_delta_errors.append(average_error(gamma, r, result[0]))
            print("Done nstep value delta")

        total_results = {"baseline" : q_var_n_step_expected_value_errors, "schedule" :  q_var_n_step_expected_value_scheduled_delta_errors, "rawvals_baseline" : vals, "rawvals_schedule": vals1}
        name = "g{:.6f}runs{:d}k{:d}ep{:.2f}".format(gamma, number_of_runs, k, epsilon)
        name = name.replace(".","")
        with open('{}.pickle'.format(name), 'wb') as handle:
            pickle.dump(total_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # Create a parser object
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--gamma", default=.95, type=float)
    parser.add_argument("--k", default=16, type=int)
    parser.add_argument("--run-index", default=None, type=int)

    # parse all arguments passed during the call
    args = parser.parse_args()

    main(args)
