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



args = get_args()
from configurations import load_params

if args.run_index is not None:
    load_params(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def main():
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
                        use_capped_bias=args.use_capped_bias)

   obs = envs.reset()
   rollouts.obs[0].copy_(obs)
   rollouts.to(device)

   episode_rewards = []
   values = []
   rewards = []
   for num_no_ops in range(30):
      really_done = False
      cur_step = 0
      while not really_done:
         with torch.no_grad():
             value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                     obs,
                     rollouts.recurrent_hidden_states[0],
                     rollouts.masks[0])

         if cur_step <= num_no_ops:
            obs, reward, done, infos = envs.step(torch.zeros((1, 1)))
         else:
            # Sample actions

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

         if num_no_ops == 0:
            if device == 'cpu':
               rewards.append(reward.numpy())
               values.append(value.numpy())
            else:
               rewards.append(reward.cpu().numpy())
               values.append(value.cpu().numpy())

         if 'episode' in infos[0].keys():
            really_done = True
            episode_rewards.append(infos[0]['episode']['r'])


         # If done then clean the history of observations.
         masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                    for done_ in done])

         rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)
         cur_step += 1

   
   with open(args.log_dir + '/random_rewards.pkl', 'wb') as handle:
      pickle.dump(episode_rewards, handle)

   with open(args.log_dir + '/values_timestep.pkl', 'wb') as handle:
      pickle.dump(values, handle)

   with open(args.log_dir + '/rewards_timestep.pkl', 'wb') as handle:
      pickle.dump(rewards, handle)
        

if __name__ == "__main__":
    main()
