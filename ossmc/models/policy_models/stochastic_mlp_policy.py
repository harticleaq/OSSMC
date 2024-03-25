import torch
import torch.nn as nn
from ossmc.utils.envs_tools import check, get_shape_from_obs_space
from ossmc.models.base.cnn import CNNBase
from ossmc.models.base.mlp import MLPBase
from ossmc.models.base.act import ACTLayer



class StochasticMlpPolicy(nn.Module):
    """Stochastic policy model that only uses MLP network. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticMlpPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticMlpPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        
        # message sender
        self.n_actions = action_space.n
        self.use_comm = self.args['use_comm']
        activation_func = nn.LeakyReLU()
        self.embed_net = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]),
            # nn.BatchNorm1d(self.hidden_sizes[-1]),
            activation_func,
            nn.Linear(self.hidden_sizes[-1],  self.hidden_sizes[-1] * 2)
        )
        self.msg_net = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1] + self.hidden_sizes[-1], self.hidden_sizes[-1]),
            activation_func,
            nn.Linear(self.hidden_sizes[-1], self.n_actions)
        )
        self.latent_dim = self.hidden_sizes[-1]
        self.var_floor = 0.1

        if self.use_comm:
            obs_dim = obs_shape + self.n_actions
        else:
            obs_dim = obs_shape


        self.base = MLPBase(args, obs_dim)


        act_dim = self.hidden_sizes[-1]


        self.act = ACTLayer(
            action_space,
            act_dim,
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(device)

    def forward(self, obs, available_actions=None, stochastic=True, messages=None):
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

        if self.use_comm:
            obs = torch.cat([obs, messages.sum(dim=1)])

        deterministic = not stochastic
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic, 
        )

        # generate message
        if self.use_comm:
            latent_parameters = self.embed_net(actor_features)
            latent_parameters[:, -self.latent_dim:] = torch.clamp(
                torch.exp(latent_parameters[:, -self.latent_dim:]),
                min=self.var_floor)
        
            latent_embed = latent_parameters
            gaussion_embed = torch.distributions.Normal(latent_embed[:, :self.latent_dim],
                                    (latent_embed[:, self.latent_dim:]) ** (1 / 2))
            latent = gaussion_embed.rsample()
            msg = self.msg_net(torch.cat([obs, latent], dim=-1))
            
        else:
            msg = actions.clone()

        return actions, msg

    def get_logits(self, obs, available_actions=None, messages=None):
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
        
        # comm
        if self.use_comm:
            obs = torch.cat([obs, messages.sum(dim=1)], dim=-1)
        
        actor_features = self.base(obs)
        return self.act.get_logits(actor_features, available_actions)
