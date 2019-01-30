# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# coding: utf-8
import pickle
import matplotlib.pyplot as  plt

MES = [
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

for game in MES:

    with open('./values/{}NoFrameskip-v4_1153780_ppoDeltaGammaGaeCappedBiasGaeValue/values_timestep.pkl'.format(game), 'rb') as f:
        x = pickle.load(f)

    fig = plt.figure(figsize=(24, 6))
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                    xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    for i in range(len(x[0][0])):
        ax1.plot([z[0][i] for z in x], label="W{}".format(i))
    
    with open('./values/{}NoFrameskip-v4_1153780_ppoDeltaGammaGaeCappedBiasGaeValue/rewards_timestep.pkl'.format(game), 'rb') as f:
        r = pickle.load(f)
        

    for i in range(len(r[0][0])):
        ax2.stem([z[0][i] for z in r], markerfmt=' ')
    
    ax1.set_ylabel("Value")
    ax2.set_ylabel("Reward")
    ax2.set_xlabel("Timestep")

    st = fig.suptitle(game, fontsize="large")
    
    plt.savefig('{}.png'.format(game), bbox_inches='tight')
    plt.close()  