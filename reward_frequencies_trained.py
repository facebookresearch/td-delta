# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import glob
import os
import time
import csv
import json
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from utils import get_vec_normalize
from visualize import visdom_plot
import pickle 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')


args = get_args()
from configurations import load_params

if args.run_index is not None:
    load_params(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def gen_frequencies():
   torch.set_num_threads(1)
   device = torch.device("cuda:0" if args.cuda else "cpu")

   envs = make_vec_envs(args.env_name, args.seed, 1,
                        args.gammas[-1], None, args.add_timestep, device, False)

   actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'num_values' : args.num_values, 'sum_values' : args.sum_values})

   state_dict = torch.load(args.log_dir + '/ppo/' + args.env_name + '.pt')
   actor_critic.load_state_dict(state_dict[0].state_dict())
   actor_critic.to(device)

   rollouts = RolloutStorage(1, 1,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size, tau=args.tau,
                        gammas=args.gammas, use_delta_gamma=args.use_delta_gamma, 
                        use_capped_bias=args.use_capped_bias, use_gae_for_value=args.use_gae_for_value)

   obs = envs.reset()
   rollouts.obs[0].copy_(obs)
   rollouts.to(device)

   episode_rewards = []
   values = []
   rewards = []
   NUM_STEPS = 10000
   total_num_rewards = 0
   for step in range(NUM_STEPS):
     
      with torch.no_grad():
         value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                  obs,
                  rollouts.recurrent_hidden_states[0],
                  rollouts.masks[0])

         
         obs, reward, done, infos = envs.step(action)

         r = reward.item()
         if r > 0 or r < 0:
            total_num_rewards += 1


         # If done then clean the history of observations.
         masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                    for done_ in done])

         rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

   
   with open('learned_frequencies/' + args.env_name[:-14] +'_learned_reward_frequency.pkl', 'wb') as handle:
      pickle.dump(total_num_rewards / NUM_STEPS, handle)


GAMES = [
         'Frostbite',
         'Alien',
         'Amidar',
         'BankHeist',
         'Hero',
         'MsPacman',
         'Qbert',
         'WizardOfWor',
         'Zaxxon',
         
         ]


SPARSE_GAMES = [
         'Freeway',
         'Gravitar',
         'MontezumaRevenge',
         'Pitfall',
         'PrivateEye',
         'Solaris',
         'Venture'
         ]
GAMES = [g + 'NoFrameskip-v4' for g in GAMES]
def plot_frequencies():
   frequencies = {}
   for game in GAMES:
      with open('learned_frequencies/' + game[:-14] + '_learned_reward_frequency.pkl', 'rb') as handle:
         frequency = pickle.load(handle)
         frequencies[game] = frequency
   sorted_keys = sorted(frequencies, key=frequencies.get)
   xs = []
   ys = []
   for game in sorted_keys:
      game_name = game[:-14]
      if game_name not in SPARSE_GAMES:
         xs.append(game_name)
         ys.append(frequencies[game] * 100)
   xticks = [i for i in range(len(ys))]
   # set font
   plt.rcParams['font.family'] = 'sans-serif'
   plt.rcParams['font.sans-serif'] = 'Helvetica'
   # set the style of the axes and the text color
   plt.rcParams['axes.edgecolor']='#333F4B'
   plt.rcParams['axes.linewidth']=0.8
   plt.rcParams['xtick.color']='#333F4B'
   plt.rcParams['ytick.color']='#333F4B'
   plt.rcParams['text.color']='#333F4B'
   fig, ax = plt.subplots(figsize=(8,3.5))
   # create for each expense type an horizontal line that starts at x = 0 with the length 
   # represented by the specific expense percentage value.
   plt.hlines(y=xs, xmin=0, xmax=ys, color='#007ACC', alpha=0.2, linewidth=5)
   # create for each expense type a dot at the level of the expense percentage value
   plt.plot(ys, xs, "o", markersize=5, color='#007ACC', alpha=0.6)
   # set labels
   ax.set_xlabel('Non-zero rewards per 100 timesteps under learned policy', fontsize=15, fontweight='black', color = '#333F4B')
   ax.set_ylabel('')
   # set axis
   ax.tick_params(axis='both', which='major', labelsize=12)
   plt.yticks(xticks, xs)
   # add an horizonal label for the y axis 
   #fig.text(-0.23, 0.96, 'Transaction Type', fontsize=15, fontweight='black', color = '#333F4B')
   # change the style of the axis spines
   ax.spines['top'].set_color('none')
   ax.spines['right'].set_color('none')
   ax.spines['left'].set_smart_bounds(True)
   ax.spines['bottom'].set_smart_bounds(True)
   # set the spines position
   ax.spines['bottom'].set_position(('axes', -0.04))
   ax.spines['left'].set_position(('axes', 0.015))
   plt.savefig('learned_frequencies.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    plot_frequencies()
