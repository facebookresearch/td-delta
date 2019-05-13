# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mdp_matrix import GridWorld, RingWorld
from value_iteration_matrix import ValueIteration, GaussSeidelValueIteration, JacobiValueIteration, PrioritizedSweepingValueIteration, GaussSeidelJacobiValueIteration
import matplotlib.pyplot as plt
import numpy as np
import pickle

gw = RingWorld()



PRECOMPUTED_VALS = {}

gammas = [.75, .875, .9375, .96875, .984375, .992, .996]


for gamma in gammas:
    vl = ValueIteration(gw)
    optimal_policy, optimal_value, _  = vl.run(gamma=gamma)
    PRECOMPUTED_VALS[gamma]  = optimal_value

with open('precomputed_vals.pickle', 'wb') as handle:
    pickle.dump(PRECOMPUTED_VALS, handle, protocol=pickle.HIGHEST_PROTOCOL)

