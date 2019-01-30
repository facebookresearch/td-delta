# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import numpy as np
import pickle
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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



def gen_frequencies():
   NUM_STEPS = 10000
   results = {}
   for game in GAMES:
      print()
      env = gym.make(game)
      is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
      if is_atari:
         env = make_atari(game) 
      env = wrap_deepmind(env)
      num_actions = env.action_space.n
      
      total_num_rewards = 0
      env.reset()
      for step in range(NUM_STEPS):
         

         
         obs, reward, done, info = env.step(np.random.randint(num_actions))
         if step == 0:
            print(info)
            print()
         print(game, step, end="\r")
         if reward > 0 or reward < 0:
            total_num_rewards += 1


         if info['ale.lives'] == 0:
            env.reset()
         


      print(game, total_num_rewards / NUM_STEPS)
      results[game] = total_num_rewards / NUM_STEPS


   with open('frequency.pkl', 'wb') as handle:
       pickle.dump(results, handle)



def plot_frequencies():
   with open('frequency.pkl', 'rb') as handle:
    frequencies = pickle.load(handle)
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
   ax.set_xlabel('Non-zero rewards per 100 timesteps under random policy', fontsize=15, fontweight='black', color = '#333F4B')
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
   plt.savefig('frequencies.png', bbox_inches='tight', dpi=300)



gen_frequencies()