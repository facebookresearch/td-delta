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
plt.switch_backend('agg')
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
from visdom import Visdom



def average_error(gamma, seed, vals):
    real_vals = PRECOMPUTED_VALS[gamma][seed]
    value_errors = []
    for val in vals:
        errors = []
        for state, value in enumerate(val):
            errors.append(np.abs(real_vals[state] - value))
        value_errors.append(np.mean(errors))
    return value_errors
plt.rcParams['text.usetex'] = True


colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]


def main(args):
    viz = Visdom(port=8097)
    number_of_runs = 200
    epsilon = .1
    ITERATIONS = [10, 25, 50, 100, 200, 500, 1000, 5000, 10000]

    alphas = [x for x in np.arange(0.0, 1., .1)]
    alphas[0] = .01
    for gamma in [.75, .875, .9375, .96875, .984375, .992, .996]:
        for k in [4, 8,  16, 32, 64, 128, 256]:
            q_var_n_step_expected_value_errors = np.empty((len(ITERATIONS), number_of_runs, len(alphas)))
            q_var_n_step_expected_value_scheduled_delta_errors = np.empty((len(ITERATIONS), number_of_runs, len(alphas)))

            # print test_rewards
            results = []
            for a, alpha in enumerate(alphas):
                for run in range(number_of_runs):
                    name = "g{:.6f}alpha{:.3f}run{:d}k{:d}ep{:.2f}".format(gamma, alpha, run, k, epsilon)
                    name = name.replace(".","")
                    with open('{}.pickle'.format(name), 'rb') as handle:
                        result = pickle.load(handle)
                        for it, iteration in enumerate(ITERATIONS):
                            q_var_n_step_expected_value_errors[it, run, a] = sum(result["baseline"][0][:iteration]) / iteration
                            q_var_n_step_expected_value_scheduled_delta_errors[it, run, a] = sum(result["schedule"][0][:iteration]) / iteration


            for it, iteration in enumerate(ITERATIONS):

                q_var_n_step_expected_value_scheduled_delta = np.mean(q_var_n_step_expected_value_scheduled_delta_errors[it], axis = 0)
                q_var_n_step_expected_value = np.mean(q_var_n_step_expected_value_errors[it], axis = 0)

                q_var_n_step_expected_value_scheduled_delta_stderr = stats.sem(q_var_n_step_expected_value_scheduled_delta_errors[it], axis=0)
                q_var_n_step_expected_value_stderr = stats.sem(q_var_n_step_expected_value_errors[it], axis=0)

                # q_var_n_step_expected_value_scheduled_delta = np.mean(np.mean(np.split(np.array(q_var_n_step_expected_value_scheduled_delta_errors), number_of_runs), axis = 0), axis=1)
                # q_var_n_step_expected_value = np.mean(np.mean(np.split(np.array(q_var_n_step_expected_value_errors), number_of_runs), axis = 0), axis=1)

                # q_var_n_step_expected_value_scheduled_delta_stderr = stats.sem(np.mean(np.split(np.array(q_var_n_step_expected_value_scheduled_delta_errors), number_of_runs), axis=0), axis = 1)
                # q_var_n_step_expected_value_stderr = stats.sem(np.mean(np.split(np.array(q_var_n_step_expected_value_errors), number_of_runs), axis=0), axis = 1)
                fig = plt.figure(figsize=(10, 8))
                ax = plt.subplot() # Defines ax variable by creating an empty plot
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontname('Arial')
                    label.set_fontsize(28)


                plt.errorbar(alphas, q_var_n_step_expected_value_scheduled_delta, np.array(q_var_n_step_expected_value_scheduled_delta_stderr), label="K-step TD(Delta) (scheduled k)", color=colors[0])
                plt.errorbar(alphas, q_var_n_step_expected_value,  np.array(q_var_n_step_expected_value_stderr), label="K-step TD", color=colors[1])
                axis_font = {'fontname':'Arial', 'size':'32'}


                plt.ylabel('Error', **axis_font)
                plt.xlabel("$\\alpha$", **axis_font)
                ax = plt.gca()
                # ax.set_xscale('symlog')
                ax.legend(loc='upper center', prop={'size': 16})
                fig_name = "g{:.6f}runs{:d}k{:d}iter{:d}".format(gamma, number_of_runs, k, iteration)
                fig_name = name.replace(".","")
                plt.title("$\\gamma = {}, k = {}$, after n={} time-steps".format(gamma, k, iteration), **axis_font)
                plt.savefig('{}.png'.format(fig_name), bbox_inches='tight')
                plt.show()
                plt.draw()
                image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
                plt.close(fig)
                image = np.transpose(image, (2, 0, 1))
                viz.image(image)


if __name__ == '__main__':
    # Create a parser object
    parser = argparse.ArgumentParser(description="Run experiments")

    # parse all arguments passed during the call
    args = parser.parse_args()

    main(args)
