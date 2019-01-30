# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


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
GAMES = [g + 'NoFrameskip-v4' for g in GAMES]
SEEDS = [125125,
         513,
         90135,
         81212,
         3523401,
         15709,
         17,
         0,
         8412,
         1153780
         ]

final_horizon = 100
final_gamma = 1 - 1 / final_horizon

gammas = [final_gamma]
horizons = [final_horizon]
cur_horizon = final_horizon
while cur_horizon >= 4:
  cur_horizon = cur_horizon / 2
  gammas = [1 - 1 / cur_horizon] + gammas
  horizons = [cur_horizon] + horizons


#            gamma,                      name,      algo,  gae,   dg,   capped, gae_v, sum_v
GAMMAS = [
          ([final_gamma],           'Baseline',     'ppo', True,  False, False,  True, False),
          ([final_gamma],           'Baseline',     'ppo', True,  False, False,  True, True),

          (gammas,                  'DeltaGamma',   'ppo', True,  True,  False,  True, False),
          
          ([gammas[0], gammas[-1]], 'DeltaGamma3',  'ppo', True,  True,  False,  True, False),
          ([gammas[2], gammas[-1]], 'DeltaGamma12', 'ppo', True,  True,  False,  True, False), 

          ([gammas[0], gammas[-1]], 'DeltaGamma3',  'ppo', True,  True,  True,  True, False),
          ([gammas[2], gammas[-1]], 'DeltaGamma12', 'ppo', True,  True,  True,  True, False),       
          
          (gammas,                  'DeltaGamma',   'ppo', True,  True,  True,   True, False),
          ]

RUN_ID = []

for seed in SEEDS:
    for game in GAMES:
        for (gamma, name, ppo, gae, delta_gamma, capped_bias, gae_value, sum_values) in GAMMAS:
            RUN_ID.append((seed, game, ppo, gae, gamma, name, delta_gamma, capped_bias, gae_value, sum_values))


def load_params(args):
    args.seed, args.env_name, args.algo, args.use_gae, args.gammas, \
    args.name, args.use_delta_gamma, args.use_capped_bias, args.use_gae_for_value, args.sum_values = RUN_ID[args.run_index]

    if args.use_delta_gamma: # DG
      args.num_values = len(args.gammas)
    elif args.sum_values: # PPO+
      args.num_values = len(gammas)
    else: #PPO
       args.num_values = 1   

    gae_string = 'Gae' if args.use_gae else ''
    capped_bias_string = 'CappedBias' if args.use_capped_bias else ''
    gae_for_value_string = 'GaeValue' if args.use_gae_for_value else 'RegValue'
    summed_value_string = '+' if args.sum_values else ''
    args.name = args.algo + args.name + gae_string + capped_bias_string + gae_for_value_string + summed_value_string

    args.log_dir = args.log_dir + args.env_name + '_' + str(args.seed) + '_' + args.name
    args.save_dir = args.log_dir

    if args.algo == 'ppo':
      args.lr = 2.5e-4
      args.clip_param = 0.1 
      args.value_loss_coef = 1.0
      args.num_processes = 8
      args.num_steps = 128
      args.num_mini_batch = 4
      args.vis_interval = 1
      args.log_interval = 1

    print(args)

