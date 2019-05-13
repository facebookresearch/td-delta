# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

class MDP:
    def __init__(self, T, S, R, A, act_list):
        # State space
        # Integer number of states
        self.S = S

        # Transition probabilities
        # Form: np ndarray of shape (start_state, action, end_state)
        self.T = np.array(T)

        # Reward space
        # Form: vector, rewards for each state
        self.R = np.array(R)

        # Action space
        # integer, number of possible actions
        self.A = A

        # Possible actions in the MDP
        self.actions = act_list


class RingWorld(MDP):

    def __init__(self):
        S = 5
        A = 1
        act_list = [1]
        R = np.zeros((5), dtype=float)
        # Because we're transitioning into the next, we have an off by one unless we addone
        R = [0, 0.95, -.9, -0.05, 0]
        T = np.zeros((S, A, S))
        for s in range(S):
            T[s][0][(s+1)%S] = .95   
            T[s][0][s] = .05
        MDP.__init__(self, T, S, R, A, act_list)


class GridWorld(MDP):
    def __init__(self, grid_size, reward_pos):
        S = grid_size*grid_size

        R = np.zeros((grid_size, grid_size))

        # Each row of reward_pos is a tuple: x, y, reward
        for row in reward_pos:
            R[row[0], row[1]] = row[2]
        R = R.flatten()

        A = 4
        act_list = ['S', 'E', 'N', 'W']

        T = np.zeros((S, A, S))
        for start_state in range(S):
            state_i = start_state/grid_size
            state_j = (start_state)%grid_size

            # Actions indexed as: 0:S, 1:E, 2:N, 3:W
            for act in range(A):
                feas_grid = np.zeros((grid_size, grid_size))
                if(act == 0 ):
                    if(state_i+1 < grid_size):
                        feas_grid[state_i+1, state_j] = 1
                    else:
                        feas_grid[state_i, state_j] = 1

                elif(act == 1):
                    if(state_j+1 < grid_size):
                        feas_grid[state_i, state_j+1] = 1
                    else:
                        feas_grid[state_i, state_j] = 1

                elif(act == 2):
                    if(state_i-1 >= 0):
                        feas_grid[state_i-1, state_j] = 1
                    else:
                        feas_grid[state_i, state_j] = 1

                elif(act == 3):
                    if(state_j-1 >= 0):
                        feas_grid[state_i, state_j-1] = 1
                    else:
                        feas_grid[state_i, state_j] = 1


                # Flatten the feasibility grid and assign to transition matrix
                T[start_state, act, :] = feas_grid.flatten()
        MDP.__init__(self, T, S, R, A, act_list)
