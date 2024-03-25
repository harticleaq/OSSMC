import torch
import torch.nn as nn
import itertools
import numpy as np
import torch.nn.functional as F

from copy import deepcopy
from ossac.utils.envs_tools import get_shape_from_obs_space
from ossac.models.base.plain_mlp import PlainMLP
from ossac.utils.envs_tools import check


def get_combined_dim(cent_obs_feature_dim, act_spaces):
    """Get the combined dimension of central observation and individual actions."""
    combined_dim = cent_obs_feature_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            action_dims = space.nvec
            for action_dim in action_dims:
                combined_dim += action_dim
    return combined_dim



class ContinuousQNet(nn.Module):
    """Q Network for continuous and discrete action space. Outputs the q value given global states and actions.
    Note that the name ContinuousQNet emphasizes its structure that takes observations and actions as input and outputs
    the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space.
    """

    def __init__(self, args, cent_obs_space, act_spaces, device=torch.device("cpu")):
        super(ContinuousQNet, self).__init__()
        activation_func = args["activation_func"]
        hidden_sizes = args["hidden_sizes"]
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
      
        self.feature_extractor = None
        cent_obs_feature_dim = cent_obs_shape[0]
        sizes = (
            [get_combined_dim(cent_obs_feature_dim, act_spaces)]
            + list(hidden_sizes)
            + [1]
        )
        self.mlp = PlainMLP(sizes, activation_func)
        self.to(device)

    def forward(self, cent_obs, actions):
        if self.feature_extractor is not None:
            feature = self.feature_extractor(cent_obs)
        else:
            feature = cent_obs
        concat_x = torch.cat([feature, actions], dim=-1)
        q_values = self.mlp(concat_x)
        return q_values


class SoftTwinContinuousQCritic():
    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        super(SoftTwinContinuousQCritic, self).__init__(       
        )
        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.auto_alpha = args["auto_alpha"]
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=args["alpha_lr"]
            )
            self.alpha = torch.exp(self.log_alpha.detach())
        else:
            self.alpha = args["alpha"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = act_space[0].__class__.__name__
        self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
        self.critic2 = ContinuousQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        critic_params = itertools.chain(
            self.critic.parameters(), self.critic2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr,
        )
        self.turn_off_grad()

    
    def update_alpha(self, logp_actions, target_entropy):
        """Auto-tune the temperature parameter alpha."""
        log_prob = (
            torch.sum(torch.cat(logp_actions, dim=-1), dim=-1, keepdim=True)
            .detach()
            .to(**self.tpdv)
            + target_entropy
        )
        alpha_loss = -(self.log_alpha * log_prob).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha.detach())

    def get_values(self, share_obs, actions):
        """Get the soft Q values for the given observations and actions."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return torch.min(
            self.critic(share_obs, actions), self.critic2(share_obs, actions)
        )
    
    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        valid_transition,
        term,
        next_share_obs,
        next_actions,
        next_logp_actions,
        gamma,
        value_normalizer=None,
    ):

        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"

        share_obs = check(share_obs).to(**self.tpdv)

        actions = check(actions).to(**self.tpdv_a)
        one_hot_actions = []
        for agent_id in range(len(actions)):
            one_hot_action = F.one_hot(
                actions[agent_id], num_classes=self.act_space[agent_id].n
            )
            one_hot_actions.append(one_hot_action)
        actions = torch.squeeze(torch.cat(one_hot_actions, dim=-1), dim=1).to(
            **self.tpdv_a
        )
        actions = torch.tile(actions, (self.num_agents, 1))
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        valid_transition = check(np.concatenate(valid_transition, axis=0)).to(
            **self.tpdv
        )
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv_a)
        next_logp_actions = torch.sum(
            torch.cat(next_logp_actions, dim=-1), dim=-1, keepdim=True
        ).to(**self.tpdv)

        next_actions = torch.tile(next_actions, (self.num_agents, 1))
        next_logp_actions = torch.tile(next_logp_actions, (self.num_agents, 1))
        next_q_values1 = self.target_critic(next_share_obs, next_actions)
        next_q_values2 = self.target_critic2(next_share_obs, next_actions)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        if self.use_proper_time_limits:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                    check(value_normalizer.denormalize(next_q_values)).to(**self.tpdv)
                    - self.alpha * next_logp_actions
                ) * (1 - term)
                value_normalizer.update(q_targets)
                q_targets = check(value_normalizer.normalize(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (
                    next_q_values - self.alpha * next_logp_actions
                ) * (1 - term)
        else:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                    check(value_normalizer.denormalize(next_q_values)).to(**self.tpdv)
                    - self.alpha * next_logp_actions
                ) * (1 - done)
                value_normalizer.update(q_targets)
                q_targets = check(value_normalizer.normalize(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (
                    next_q_values - self.alpha * next_logp_actions
                ) * (1 - done)
        if self.use_huber_loss:
            if self.state_type == "FP" and self.use_policy_active_masks:
                critic_loss1 = (
                    torch.sum(
                        F.huber_loss(
                            self.critic(share_obs, actions),
                            q_targets,
                            delta=self.huber_delta,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
                critic_loss2 = (
                    torch.mean(
                        F.huber_loss(
                            self.critic2(share_obs, actions),
                            q_targets,
                            delta=self.huber_delta,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
            else:
                critic_loss1 = torch.mean(
                    F.huber_loss(
                        self.critic(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
                critic_loss2 = torch.mean(
                    F.huber_loss(
                        self.critic2(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
        else:
            if self.state_type == "FP" and self.use_policy_active_masks:
                critic_loss1 = (
                    torch.sum(
                        F.mse_loss(self.critic(share_obs, actions), q_targets)
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
                critic_loss2 = (
                    torch.sum(
                        F.mse_loss(self.critic2(share_obs, actions), q_targets)
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
            else:
                critic_loss1 = torch.mean(
                    F.mse_loss(self.critic(share_obs, actions), q_targets)
                )
                critic_loss2 = torch.mean(
                    F.mse_loss(self.critic2(share_obs, actions), q_targets)
                )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def turn_on_grad(self):
        """Turn on the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False

    
    def soft_update(self):
        """Soft update the target networks."""
        for param_target, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
        for param_target, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
