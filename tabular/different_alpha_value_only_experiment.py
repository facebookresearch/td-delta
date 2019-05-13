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
            errors.append((real_vals[state] - value)**2)
        value_errors.append(np.mean(errors)**(1/2))
    return value_errors

gammas = [.75, .875, .9375, .96875, .984375, .992, .996]
ks = [4, 8, 16, 32, 64, 128, 256]

# alphas = [x for x in np.arange(0.0, 1., .1)][1:]
alphas = [x for x in np.geomspace(0.001, 1.0, num=20, endpoint=False)] #+ alphas
# alphas[0] = .01
number_of_runs = 200

RUNS = []
for g in gammas:
    for k in ks:
        for a in alphas:
            RUNS.append((g, k, a))


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
        gamma, k, alpha = RUNS[args.run_index]
    else:
        gamma= args.gamma
        k =args.k
        alpha=0.1
        r=0

    epsilon = .1
    n=1

    vals = None
    vals_errors = None
    vals_scheduled_deltas = None
    vals_scheduled_delta_errors = None
    
    for r in range(number_of_runs):
        results_baseline, _, _, _, _ = n_step_expected_value(gw, n, alpha, gamma, epsilon, n=k, seed=r)
        results_delta, _, _, _, _  = n_step_expected_value_delta(gw, n, alpha, gamma, epsilon, n=k, seed=r, schedule_ks=True) 
        
        if r == 0: 
            vals = np.empty((number_of_runs, len(results_baseline), len(results_baseline[0])))
            vals_errors = np.empty((number_of_runs, len(results_baseline)))
            vals_scheduled_deltas = np.empty((number_of_runs, len(results_baseline), len(results_baseline[0])))
            vals_scheduled_delta_errors = np.empty((number_of_runs, len(results_baseline)))
            
        vals[r] = results_baseline
        vals_errors[r] = average_error(gamma, r, results_baseline)
        vals_scheduled_deltas[r] = results_delta 
        vals_scheduled_delta_errors[r] = average_error(gamma, r, results_delta)

    total_results = {"baseline" : vals_errors, "schedule" :  vals_scheduled_delta_errors, "rawvals_baseline" : vals, 
                     "rawvals_schedule": vals_scheduled_deltas}
    name = "g{:.6f}alpha{:.3f}runs{:d}k{:d}ep{:.2f}".format(gamma, alpha, number_of_runs, k, epsilon)
    name = name.replace(".","")
    with open('results_new_plus_one/{}.pickle'.format(name), 'wb') as handle:
        pickle.dump(total_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done:", args.run_index)

if __name__ == '__main__':
    # Create a parser object
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--gamma", default=.95, type=float)
    parser.add_argument("--k", default=16, type=int)
    parser.add_argument("--run-index", default=None, type=int)

    # parse all arguments passed during the call
    args = parser.parse_args()

    main(args)
