import torch
import torch.nn.functional as F

from agents.a2c_base import A2CBase
from copy import deepcopy
import numpy as np

import utils
from utils.ddpg_utils import RandomShiftsAug


class DrQv2DDPG(A2CBase):
    """
    DDPG agent class using DrQv2 augmentation methods
    """

    def __init__(self, stddev_schedule, stddev_clip, lr=0.0001, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001, target_update_interval=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau)
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.target_update_interval = target_update_interval
        self.num_update = 0
        self.aug = RandomShiftsAug(pad=4)
        self.obs_type = 'pixel'

    def _loadBatchToDevice(self, batch):
        """
        Load batch into pytorch tensor and adds DrQv2 augmentation to observation
        :param batch: list of transitions
        :return: states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, next_obs_tensor,
                 non_final_masks, step_lefts_tensor, is_experts_tensor
        """

        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs)
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.tensor(np.stack(states)).long().to(self.device)
        obs_tensor = torch.tensor(np.stack(images)).to(self.device)
        # add augmentation to observations in batch
        obs_tensor = self.aug(obs_tensor)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack(xys)).to(self.device)
        rewards_tensor = torch.tensor(np.stack(rewards)).to(self.device)
        next_states_tensor = torch.tensor(
            np.stack(next_states)).long().to(self.device)
        next_obs_tensor = torch.tensor(np.stack(next_obs)).to(self.device)
        # augment next observations in batch
        next_obs_tensor = self.aug(next_obs_tensor)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.tensor(np.stack(dones)).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.tensor(np.stack(step_lefts)).to(self.device)
        is_experts_tensor = torch.tensor(
            np.stack(is_experts)).bool().to(self.device)

        if self.obs_type is 'pixel':
            # scale observation from int to float
            obs_tensor = obs_tensor/255*0.4
            next_obs_tensor = next_obs_tensor/255*0.4

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = obs_tensor
        self.loss_calc_dict['action_idx'] = action_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = next_obs_tensor
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor

        return states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, \
            next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor

    def initNetwork(self, actor, critic, initialize_target=True):
        """
        Initialize networks
        :param actor: actor network
        :param critic: critic network
        :param initialize_target: whether to create target networks
        """
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr[0])
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr[1])
        if initialize_target:
            self.critic_target = deepcopy(critic)
            self.target_networks.append(self.critic_target)
        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)

    def calcCriticLoss(self, step):
        """
        Calculate critic loss
        :param step: current step for scheduling stddev
        Return target q1, q2, and td error
        """
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            # TODO ensure that sampling adds noise
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q1 = target_Q1.reshape(batch_size)
            target_Q2 = target_Q2.reshape(batch_size)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = rewards + non_final_masks * self.gamma * target_V

        q1, q2 = self.critic(obs, action)
        q1 = q1.reshape(batch_size)
        q2 = q2.reshape(batch_size)
        q1_loss = F.mse_loss(q1, target_Q)
        q2_loss = F.mse_loss(q2, target_Q)
        with torch.no_grad():
            # TODO see difference in wrapping 0.5 * around entire expression
            td_error = 0.5 * (torch.abs(q2 - target_Q) +
                              torch.abs(q1 - target_Q))

        return q1_loss, q2_loss, td_error

    def updateCritic(self, step):
        """Updates both the critic networks

        Returns:
            3 torch.Tensors: q1 loss, q2 loss, and td error (not critic loss)
        """
        # TODO: add metrics dictionary
        q1_loss, q2_loss, td_error = self.calcCriticLoss(step)

        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        # encoder is also optimized with critic optimizer backward pass
        critic_loss.backward()
        self.critic_optimizer.step()

        return q1_loss, q2_loss, td_error

    def calcActorLoss(self, step):
        """Calculates loss for actor network

        Args:
            step (int): current step in training

        Returns:
            tensor: actor loss
        """
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        # log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        q1, q2 = self.critic(obs, action)
        q = torch.min(q1, q2)

        return -q.mean()

    def updateActor(self, step):
        """Updates the actor for one step

        Args:
            step (int): current step in training

        Returns:
            tensor: actor loss
        """
        actor_loss = self.calcActorLoss(step)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss

    def update(self, batch):
        # overriden to load augmented states
        self._loadBatchToDevice(batch)

        # TODO: verify that self.numUpdate is indeed step
        q1_loss, q2_loss, td_error = self.updateCritic(self.num_update)
        actor_loss = self.updateActor(self.num_update)

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        return q1_loss.item(), q2_loss.item(), td_error.item(), actor_loss.item()
