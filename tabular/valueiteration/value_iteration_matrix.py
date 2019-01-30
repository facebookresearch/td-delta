# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from operator import itemgetter
from pqdict import pqdict

class ValueIteration(object):

    def __init__(self, mdp, gauss_seidel=False):
        self.mdp = mdp
        self.gauss_seidel = gauss_seidel

    def run(self, theta = 0.001, gamma=.9, optimal_value=None):
        # initialize array V arbitrarily
        # V(s) = 0 for s in S
        V = np.zeros(self.mdp.S)
        vs = []
        iteration = 0
        sweeps = 0
        v0 = None
        v1 = None
        printed_v0v1 = False
        while True:
            delta = 0
            if self.gauss_seidel:
                # as per slides http://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2016/04/02-MarkovDecisionProcess.pdf
                # simply allow updates to the current state-value space
                Vold = V
            else:
                Vold = V.copy()

            for s in range(self.mdp.S):
                if optimal_value is not None:
                    vs.append(np.linalg.norm(V - optimal_value))
                iteration += 1
                v = Vold[s]


                possibilities = []
                for a in range(self.mdp.A):
                    possibilities.append((self.mdp.R[s] + gamma * sum(self.mdp.T[s,a,k] * Vold[k] for k in range(self.mdp.S))))
                V[s] = max(possibilities)

                # Sutton, p.90 2nd edition draft (Jan. 2017)

                delta = max(delta, abs(v - V[s]))

            sweeps += 1
            if v0 is None:
                v0 = V.copy()
            elif v1 is None:
                v1 = V.copy()
            elif not printed_v0v1:
                print("||V1 - V0|| = %f" % np.linalg.norm(v1 - v0))
                printed_v0v1 = True

            if delta < theta:
                break

        print("Converged in %d iterations (%d sweeps)" % (iteration, sweeps))

        pi = self.get_policy(V)

        return pi, V, vs

    def get_policy(self, V, gamma=0.9):
        pi = {}
        for s in range(self.mdp.S):
            possibilities = [sum(self.mdp.T[s,a,k] *(self.mdp.R[k] + gamma * V[k]) for k in range(self.mdp.S)) for a in range(self.mdp.A)]
            pi[s] = max(enumerate(possibilities), key=itemgetter(1))[0]

        return pi

class GaussSeidelValueIteration(ValueIteration):

    def __init__(self, mdp):
        super(GaussSeidelValueIteration, self).__init__(mdp, gauss_seidel=True)

class JacobiValueIteration(ValueIteration):

    def run(self, theta = 0.01, gamma=.9, optimal_value=None):
        # initialize array V arbitrarily
        # print("Jacobian")
        # V(s) = 0 for s in S
        V = np.zeros(self.mdp.S)
        vs = []

        iteration = 0
        sweeps = 0
        while True:
            delta = 0
            if self.gauss_seidel:
                # as per slides http://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2016/04/02-MarkovDecisionProcess.pdf
                # simply allow updates to the current state-value space
                Vold = V
            else:
                Vold = V.copy()

            for s in range(self.mdp.S):
                iteration += 1
                if optimal_value is not None:
                    vs.append(np.linalg.norm(V - optimal_value))
                v = Vold[s]
                # As in https://tspace.library.utoronto.ca/bitstream/1807/24381/6/Shlakhter_Oleksandr_201003_PhD_thesis.pdf
                possibilities = []

                for a in range(self.mdp.A):
                    possibilities.append((self.mdp.R[s] + gamma * sum(self.mdp.T[s,a,k] * Vold[k] for k in range(self.mdp.S) if k != s)) /  (1. - gamma * self.mdp.T[s][a][s]))
                V[s] = max(possibilities)

                # Sutton, p.90 2nd edition draft (Jan. 2017)
                delta = max(delta, abs(v - V[s]))
            sweeps += 1
            if delta < theta:
                break

        print("Converged in %d iterations (%d sweeps)" % (iteration, sweeps))

        pi = self.get_policy(V)

        return pi, V, vs

class GaussSeidelJacobiValueIteration(JacobiValueIteration):
    def __init__(self, mdp):
        super(GaussSeidelJacobiValueIteration, self).__init__(mdp, gauss_seidel=True)

class PrioritizedSweepingValueIteration(ValueIteration):

    def run(self, theta=0.0001, gamma=.9, max_iterations= 2000, optimal_value = None):
        # as per slides http://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2016/04/02-MarkovDecisionProcess.pdf
        # and http://www.jmlr.org/papers/volume6/wingate05a/wingate05a.pdf
        V = np.zeros(self.mdp.S)
        H = np.zeros(self.mdp.S)
        vs = []
        iterations = 0

        predecessors = {}
        for state in range(self.mdp.S):
          predecessors[state] = set()

        priority_queue = pqdict()

        vs = []

        for state in range(self.mdp.S):
            for a in range(self.mdp.A):
                for index, s in enumerate(self.mdp.T[state, a]):
                    if s > 0:
                        predecessors[index].add(state)

            possibilities = []
            v = V[state]
            for a in range(self.mdp.A):
                possibilities.append((self.mdp.R[state] + gamma * sum(self.mdp.T[state,a,k] * V[k] for k in range(self.mdp.S))))
            prob = max(possibilities)

            delta = abs(v - prob)
            priority_queue[state] = -delta

        for i in range(max_iterations):
            if len(priority_queue) == 0:
                break
            if optimal_value is not None:
                vs.append(np.linalg.norm(V - optimal_value))
            state = priority_queue.pop()

            # update V[state]
            v = V[state]
            Vold = V.copy()
            possibilities = []
            for a in range(self.mdp.A):
                possibilities.append((self.mdp.R[state] + gamma * sum(self.mdp.T[state,a,k] * Vold[k] for k in range(self.mdp.S))))
            V[state] = max(possibilities)

            # Update all predecessors priorities
            for p in predecessors[state]:
                v = V[p]
                possibilities = []
                for a in range(self.mdp.A):
                    possibilities.append((self.mdp.R[p] + gamma * sum(self.mdp.T[p,a,k] * V[k] for k in range(self.mdp.S))))
                priority = max(possibilities)

                delta = abs(v - priority)

                if delta > theta:
                    priority_queue[p] = -delta



        print("Converged in %d iterations" % (i))

        pi = self.get_policy(V)

        return pi, V, vs
