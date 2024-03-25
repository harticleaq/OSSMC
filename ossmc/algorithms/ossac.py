import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from ossmc.utils.envs_tools import check, get_shape_from_obs_space
from ossmc.utils.models_tools import init, get_active_func, get_init_method
from ossmc.models.base.distributions import Categorical
from ossmc.utils.discrete_util import gumbel_softmax

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)



class MLPBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_dim = obs_shape

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func
        )

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x


class ACTLayer(nn.Module):
    def __init__(
            self, action_space, inputs_dim, initialization_method, gain, args=None
        ):
            """Initialize ACTLayer.
            Args:
                action_space: (gym.Space) action space.
                inputs_dim: (int) dimension of network input.
                initialization_method: (str) initialization method.
                gain: (float) gain of the output layer of the network.
                args: (dict) arguments relevant to the network.
            """
            super(ACTLayer, self).__init__()
            self.action_type = action_space.__class__.__name__

            action_dim = action_space.n
            self.action_out = Categorical(
                inputs_dim, action_dim, initialization_method, gain
            )
            self.use_comm = args['use_comm']


    def forward(self, x, available_actions=None, deterministic=False):
        """Compute actions and action logprobs from given input.
        Args:
            x: (torch.Tensor) input to network.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """

        action_distribution = self.action_out(x, available_actions)
        actions = (
            action_distribution.mode()
            if deterministic
            else action_distribution.sample()
        )
        action_log_probs = action_distribution.log_probs(actions)

        return actions, action_log_probs

    def get_logits(self, x, available_actions=None):

        action_distribution = self.action_out(x, available_actions)
        action_logits = action_distribution.logits
        return action_logits

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_distribution = self.action_out(x, available_actions)
        action_log_probs = action_distribution.log_probs(action)
        if active_masks is not None:
            dist_entropy = (
                action_distribution.entropy() * active_masks.squeeze(-1)
            ).sum() / active_masks.sum()
  
        else:
            dist_entropy = action_distribution.entropy().mean()

        return action_log_probs, dist_entropy, action_distribution



class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()

        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        # Message sender
        self.var_floor = 0.002
        self.n_actions = action_space.n
        self.latent_dim = self.n_actions
        self.use_comm = self.args['use_comm']
        activation_func = [nn.LeakyReLU(), nn.ReLU()][0]
        self.embed_net = nn.Sequential(
        nn.Linear(self.hidden_sizes[-1]+self.latent_dim, self.hidden_sizes[-1]),
        # nn.BatchNorm1d(self.hidden_sizes[-1]),
        activation_func,
        nn.Linear(self.hidden_sizes[-1], self.latent_dim * 2)
        )
        self.msg_net = nn.Sequential(
            nn.Linear(obs_shape[0], self.hidden_sizes[-1]),
            activation_func,
            nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1])
        )
        self.inference_net = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1] + self.n_actions + self.latent_dim, self.hidden_sizes[-1] ),
            activation_func,
            nn.Linear(self.hidden_sizes[-1], self.latent_dim * 2)
        )

        if self.use_comm:
            obs_dim = obs_shape[0] 
        else:
            obs_dim = obs_shape[0]
        self.base = MLPBase(args, obs_dim)
        act_dim = self.hidden_sizes[-1] + self.latent_dim
        self.act = ACTLayer(
            action_space,
            act_dim,
            self.initialization_method,
            self.gain,
            args,
        )

        # Message receiver
        self.attention_dim = self.hidden_sizes[-1]
        self.w_query = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], self.attention_dim),
            activation_func,
            nn.Linear(self.attention_dim, self.n_actions)
        )
        self.w_key = nn.Sequential(
            nn.Linear(self.n_actions, self.attention_dim),
            activation_func,
            nn.Linear(self.attention_dim, self.n_actions)
        )
        self.w_value =  nn.Sequential(
            nn.Linear(self.n_actions, self.attention_dim),
            activation_func,
            nn.Linear(self.attention_dim, self.n_actions)
        )

        self.to(device)


    def forward(self, obs, available_actions=None, stochastic=True, messages=None, agent_id=None):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            stochastic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
        """
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        # Receive messages
        if self.use_comm:
            personal_messages, alpha = self.compute_messages(actor_features, messages, agent_id)
            actor_features = torch.cat([actor_features, personal_messages], dim=-1)

        
   
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic,  
        )

        # Generate message i
        if self.use_comm:
            # comm_features = self.msg_net(obs_origin)
            latent_parameters = self.embed_net(actor_features)
            latent_parameters[:, -self.latent_dim:] = torch.clamp(
                torch.exp(latent_parameters[:, -self.latent_dim:]),
                min=self.var_floor)
        
            latent_embed = latent_parameters
            gaussion_embed = torch.distributions.Normal(latent_embed[:, :self.latent_dim],
                                    (latent_embed[:, self.latent_dim:]) ** (1 / 2))
            if stochastic:
                latent = gaussion_embed.rsample()
            else:
                latent = gaussion_embed.mode
            msg = latent.clone()
            # msg[available_actions == 0] = 0
            mi_loss = self.compute_im_loss(gaussion_embed, actions, actor_features)
        else:
            msg = actions.clone()
            mi_loss = None
            alpha = None

        return actions, msg, mi_loss, alpha

    # Motivational attention messages
    def compute_messages(self, obs, messages, agent_id):
        q = self.w_query(obs).view(obs.shape[0], 1, -1)
        k = self.w_key(messages).transpose(1, 2)
        v = self.w_value(messages)
        alpha = torch.bmm(q / (self.attention_dim ** (1/2)), k)
        # alpha[:, :, agent_id] = -1e9
        alpha = F.softmax(alpha, dim=-1)
        messages = alpha @ v
        messages = messages.squeeze(dim=-2)
        return messages, alpha
    
    # Mi loss
    def compute_im_loss(self, latent_embed, actions, obs):
        one_hot_action = check(torch.zeros(actions.shape[0], self.n_actions)).to(**self.tpdv).scatter(1, actions, 1)
        
        latent_infer = self.inference_net(torch.cat([obs, one_hot_action], dim=-1))
        latent_infer[:, self.latent_dim:] = torch.clamp(torch.exp(latent_infer[:, self.latent_dim:]), min=self.var_floor)
        g2 = D.Normal(latent_infer[:, :self.latent_dim], latent_infer[:, self.latent_dim:] ** (1/2))

        mi_loss = D.kl_divergence(latent_embed, g2).sum(-1).mean()
        return mi_loss * self.args["mi_loss_weight"]


    def get_logits(self, obs, available_actions=None, messages=None, agent_id=None):
        """Get action logits from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) input to network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                      (if None, all actions available)
        Returns:
            action_logits: (torch.Tensor) logits of actions for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        actor_features = self.base(obs)
        # Compute personal messages
        if self.use_comm:
            messages, _ = self.compute_messages(actor_features, messages, agent_id)
            actor_features = torch.cat([actor_features, messages], dim=-1)

        
        return self.act.get_logits(actor_features, available_actions)


class OSSAC:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.action_type = act_space.__class__.__name__

        self.actor = Actor(args, obs_space, act_space, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.turn_off_grad()
        self.use_comm = args['use_comm']


    def get_actions(self, obs, available_actions=None, stochastic=True, msgs=None, agent_id=None):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        if self.use_comm:
            msgs = check(msgs).to(**self.tpdv)
        actions, msg, mi_loss, alpha = self.actor(obs, available_actions, 
              stochastic, messages=msgs, agent_id=agent_id)
        return actions, msg, mi_loss, alpha
    
    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True, messages=None, agent_id=None):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        if self.use_comm:
            messages = check(messages).to(**self.tpdv)
        logits = self.actor.get_logits(obs, available_actions, messages=messages, agent_id=agent_id)
        actions = gumbel_softmax(
            logits, hard=True, device=self.device
        )  # onehot actions
        logp_actions = torch.sum(actions * logits, dim=-1, keepdim=True)
        return actions, logp_actions
    


    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = True
 

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = False


    def soft_update(self):
        """Soft update target actor."""
        for param_target, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def save(self, save_dir, id):
        """Save the actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "\\actor_agent" + str(id) + ".pt"
        )

    def restore(self, model_dir, id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "\\actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)



