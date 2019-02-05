# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size, 
        tau=None, gammas=None, use_delta_gamma=False, use_capped_bias=False):
        assert gammas is not None and tau is not None

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, len(gammas))
        self.value_preds = torch.zeros(num_steps + 1, num_processes, len(gammas))
        self.returns = torch.zeros(num_steps + 1, num_processes, len(gammas))
        self.policy_returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0

        self.use_delta_gamma = use_delta_gamma
        self.use_capped_bias = use_capped_bias
        self.gamma_list = gammas
        self.gammas = torch.from_numpy(np.array(gammas)).type(self.rewards.type()).unsqueeze(0)

        self.zeros = torch.zeros(num_processes, len(gammas))

        self.unbiased_taus = [(tau * gammas[-1]) / g for g in gammas]

        if self.use_capped_bias:
            horizons = [int(np.ceil((1/(1-g)))) for g in gammas]
            taus = [min(1.0, (tau * gammas[-1]) / g) for g in gammas]
        else:
            horizons = [int(np.ceil((1/(1-gammas[-1])))) for _ in gammas]
            taus = self.unbiased_taus

        self.horizons = horizons
        self.starting_taus = taus
        self.taus = torch.from_numpy(np.array(taus)).type(self.rewards.type()).unsqueeze(0)

        if self.use_delta_gamma:
            self.delta_rewards = torch.zeros(num_steps, num_processes, len(gammas)-1)
            self.delta_gamma_vector = torch.from_numpy(np.array([gammas[i] - gammas[i - 1] 
                                    for i in range(1, len(gammas))])).type(self.rewards.type()).unsqueeze(0)
            

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.policy_returns = self.policy_returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.gammas = self.gammas.to(device)
        self.taus = self.taus.to(device)
        self.zeros = self.zeros.to(device)

        if self.use_delta_gamma:
            self.delta_rewards = self.delta_rewards.to(device)
            self.delta_gamma_vector = self.delta_gamma_vector.to(device)
            


    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.masks[self.step + 1].copy_(masks)
        self.rewards[self.step, :, 0:1].copy_(rewards)

        if self.use_delta_gamma and self.step > 0:
            self.delta_rewards[self.step-1] = self.compute_delta_rewards(self.rewards[self.step-1], 
                                                                         self.value_preds[self.step-1], 
                                                                         self.value_preds[self.step], 
                                                                         self.masks[self.step],
                                                                         self.rewards[self.step]) 
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        

    def compute_returns(self, next_value, use_gae, gamma, tau):
        self.value_preds[-1] = next_value

        if self.use_delta_gamma:
            # delta rewards stores the delta gamma of the previous value function at the next time step
            self.delta_rewards[-1] = self.compute_delta_rewards(self.rewards[-1], 
                                                                self.value_preds[-1], 
                                                                self.value_preds[-2], 
                                                                self.masks[-1],
                                                                self.zeros)   
            self.rewards = torch.cat([self.rewards[:, :, 0].unsqueeze(-1), self.delta_rewards], dim=-1)
        else:
            # store real rewards in every value function slot
            self.rewards = self.rewards[:, :, 0].unsqueeze(-1).expand_as(self.rewards)

        # policy_returns will store return to be used by policy
        # returns will store return for each value function
        self.returns[-1] = next_value
        self.policy_returns[-1] = next_value.sum(-1).unsqueeze(-1)
        gae = 0
        gae_value = 0
        for step in reversed(range(self.rewards.size(0))):
            if use_gae:
                delta_value = self.rewards[step] + self.gammas.expand_as(self.rewards[step]) * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae_value = delta_value + self.gammas.expand_as(self.rewards[step]) * self.taus.expand_as(self.rewards[step]) * self.masks[step + 1] * gae_value
                self.returns[step] = gae_value + self.value_preds[step]

                delta = self.rewards[step, :, 0].unsqueeze(-1) + gamma * self.value_preds[step + 1].sum(-1).unsqueeze(-1) * self.masks[step + 1] - self.value_preds[step].sum(-1).unsqueeze(-1)
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.policy_returns[step] = gae + self.value_preds[step].sum(-1).unsqueeze(-1) 

            else:
                self.returns[step] = self.returns[step + 1] * \
                    self.gammas.expand_as(next_value) * self.masks[step + 1].expand_as(self.rewards[step]) + self.rewards[step]
                self.policy_returns[step] = self.policy_returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step, :, 0].unsqueeze(-1)

                    
    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, len(self.gamma_list))[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, len(self.gamma_list))[indices]
            
            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


    def compute_delta_rewards(self, rewards, v, v_prime, masks, next_rewards):
        # the value for each gamma is the sum of all delta gammas up to the current gamma
        v_gamma_prime = torch.cumsum(v_prime, dim=-1)[:, :-1] 
        v_gamma = torch.cumsum(v, dim=-1)[:, :-1]

        delta_rewards = masks.expand_as(v_gamma) * self.delta_gamma_vector.expand_as(v_gamma) * v_gamma_prime

        return delta_rewards

        




