import os
import numpy as np
import torch
import torch.nn.functional as F
import setproctitle
import pandas as pd

from ossmc.common.valuenorm import ValueNorm
from torch.distributions import Categorical
from ossmc.utils.trans_tools import _t2n
from ossmc.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from ossmc.utils.models_tools import init_device
from ossmc.algorithms.ossac import OSSAC as Policy
from ossmc.algorithms.critics.soft_twin_continuous_q_critic import SoftTwinContinuousQCritic
from ossmc.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
from ossmc.utils.configs_tools import init_dir, save_config, get_task_name

class Runner:
    def __init__(self, args, algo_args, env_args):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.episode_beta = []
        self.episode_messages = []

        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = self.algo_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.task_name = get_task_name(args["env"], env_args)
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])

        if (
            "use_valuenorm" in self.algo_args["train"].keys()
            and self.algo_args["train"]["use_valuenorm"]
        ):
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
        self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
        self.eval_envs = (
            make_eval_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["eval"]["n_eval_rollout_threads"],
                env_args,
            )
            if algo_args["eval"]["use_eval"]
            else None
                )
        
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_deaths = np.zeros(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
        )

        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)

        self.actor = []
        for agent_id in range(self.num_agents):
                agent = Policy(
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        self.critic = SoftTwinContinuousQCritic(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                self.envs.share_observation_space[0],
                self.envs.action_space,
                self.num_agents,
                self.state_type,
                device=self.device,
            )
        
        self.buffer = OffPolicyBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
        

        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()

        self.total_it = 0  # total iteration

        if (
            "auto_alpha" in self.algo_args["algo"].keys()
            and self.algo_args["algo"]["auto_alpha"]
        ):
            self.target_entropy = []
            for agent_id in range(self.num_agents):
                 self.target_entropy.append(
                        -0.98
                        * np.log(1.0 / np.prod(self.envs.action_space[agent_id].shape))
                    )
            self.log_alpha = []
            self.alpha_optimizer = []
            self.alpha = []
            for agent_id in range(self.num_agents):
                _log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.log_alpha.append(_log_alpha)
                self.alpha_optimizer.append(
                    torch.optim.Adam(
                        [_log_alpha], lr=self.algo_args["algo"]["alpha_lr"]
                    )
                )
                self.alpha.append(torch.exp(_log_alpha.detach()))
        elif "alpha" in self.algo_args["algo"].keys():
            self.alpha = [self.algo_args["algo"]["alpha"]] * self.num_agents

        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
        save_config(args, algo_args, env_args, self.run_dir)
        self.log_file = open(
            os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
        )

        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )


    def train(self):
        self.use_seqrand = True
        self.use_discrete = True
        self.num_random = 7
        self.total_it += 1

        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_messages
        ) = data

        self.critic.turn_on_grad()
        next_actions = []
        next_logp_actions = []
        for agent_id in range(self.num_agents):
            next_action, next_logp_action = self.actor[
                agent_id
            ].get_actions_with_logprobs(
                sp_next_obs[agent_id],
                sp_next_available_actions[agent_id],
                messages=sp_messages, 
                agent_id=agent_id
            )
            next_actions.append(next_action)
            next_logp_actions.append(next_logp_action)
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            next_actions,
            next_logp_actions,
            sp_gamma,
            self.value_normalizer,
        )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            actions = []
            logp_actions = []
            
            with torch.no_grad():
                for agent_id in range(self.num_agents):
                    action, logp_action = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id],
                        messages=sp_messages, 
                        agent_id=agent_id
                    )
                    actions.append(action)
                    logp_actions.append(logp_action)

            if self.fixed_order:
                agent_order = list(range(self.num_agents))
            else:
                agent_order = list(np.random.permutation(self.num_agents))
            for agent_id in agent_order:
                self.actor[agent_id].turn_on_grad()
                # train this agent
                actions[agent_id], logp_actions[agent_id] = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id],
                    messages=sp_messages, 
                        agent_id=agent_id
                )
                logp_action = torch.tile(
                    logp_actions[agent_id], (self.num_agents, 1)
                )

                # Construct optimistic Q-value
                num_random = 7
                random_actions_tensor = torch.zeros(num_random, len(actions), actions[0].shape[0], sp_available_actions.shape[-1]).to(self.device)
                sp_index = torch.FloatTensor(sp_available_actions).to(self.device)
                for i in range(num_random):
                    for j in range(self.num_agents):
                        if j <= agent_id:
                            random_actions_tensor[i, j] = actions[j]
                        else:
                            random_actions_tensor[i, j] = F.one_hot(torch.multinomial(sp_index[j], 1), sp_available_actions.shape[-1]).squeeze(-2)

                random_actions_tensors = []
                for i in range(random_actions_tensor.shape[1]):
                    random_actions_tensors.append(torch.cat([actions[i].unsqueeze(0), random_actions_tensor[:, i]], 0))
                random_actions_tensors_ = torch.cat(random_actions_tensors, -1)

                random_actions_tensors_t = random_actions_tensors_.repeat(1, self.num_agents, 1) #torch.tile(random_actions_tensors_, (1, self.num_agents, 1)) 
                sp_share_obs_t = sp_share_obs.reshape(1, random_actions_tensors_t.shape[1], -1).repeat(random_actions_tensors_t.shape[0], 0)
                
                value_pred = self.critic.get_values(sp_share_obs_t, random_actions_tensors_t)
                value_pred = value_pred.max(dim=0)[0]

                # actions_t = torch.tile(
                #     torch.cat(actions, dim=-1), (self.num_agents, 1)
                #         )
                # value_pred = self.critic.get_values(sp_share_obs, actions_t)
                    
                if self.algo_args["algo"]["use_policy_active_masks"]:
                    valid_transition = torch.tile(
                        sp_valid_transition[agent_id], (self.num_agents, 1)
                    )
                    actor_loss = (
                        -torch.sum(
                            (value_pred - self.alpha[agent_id] * logp_action)
                            * valid_transition
                        )
                        / valid_transition.sum()
                    )
                else:
                    actor_loss = -torch.mean(
                            value_pred - self.alpha[agent_id] * logp_action
                        )

                # Add MI loss 
                if self.algo_args["algo"]["use_comm"]:
                    _, _, mi_loss, _ = self.actor[agent_id].get_actions(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id],
                            stochastic=False,
                            msgs=sp_messages, 
                            agent_id=agent_id
                        )
                    actor_loss += mi_loss

                self.actor[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()
                # train this agent's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    log_prob = (
                        logp_actions[agent_id].detach()
                        + self.target_entropy[agent_id]
                    )
                    alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                    self.alpha_optimizer[agent_id].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer[agent_id].step()
                    self.alpha[agent_id] = torch.exp(
                        self.log_alpha[agent_id].detach()
                    )
                actions[agent_id], _ = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id],
                    messages=sp_messages,
                        agent_id=agent_id
                )
            # train critic's alpha
            if self.algo_args["algo"]["auto_alpha"]:
                self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            self.critic.soft_update()
    
    def run(self):
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        # obs, share_obs, available_actions = self.envs.reset()
        self.messages = torch.zeros(obs.shape[0], obs.shape[1],
                                     available_actions.shape[-1]).to(self.device)
        

        print(" start training")

        steps = (
            self.algo_args["train"]["num_env_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        update_num = int(  # update number per train
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )

        for step in range(1, steps + 1):
            actions = self.get_actions(
                obs, available_actions=available_actions, add_random=True,
            )
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(
                actions
            )

            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2),
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2),
                _t2n(self.messages)
            )

            # Init messages if done
            if np.all(dones, axis=1)[0]:
                self.init_messages()
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            if step % self.algo_args["train"]["train_interval"] == 0:
                for _ in range(update_num):
                    self.train()
            if step % self.algo_args["train"]["eval_interval"] == 0:
                cur_step = (
                    self.algo_args["train"]["warmup_steps"]
                    + step * self.algo_args["train"]["n_rollout_threads"]
                )//self.algo_args["train"]["eval_interval"]
               
                print(
                    f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Evaluation at step {cur_step} / {self.algo_args['train']['num_env_steps']}:"
                )
                self.eval(cur_step)
                # self.save()

    def restore(self):
        """Restore the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].restore(self.algo_args["train"]["model_dir"], agent_id)
        if not self.algo_args["render"]["use_render"]:
            self.critic.restore(self.algo_args["train"]["model_dir"])
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)
        print("load model success!")     

    def save(self):
        """Save the model"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(self.save_dir, agent_id)
        self.critic.save(self.save_dir)
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )
    # Init messages
    def init_messages(self):
        self.messages = torch.zeros_like(self.messages).to(self.device)
        
    @torch.no_grad()
    def eval(self, step):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        eval_battles_won = 0

        episode_lens = []
        episode_beta = []
        episode_messages = []
        one_episode_len = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"], dtype=np.int
        )

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        while True:
            eval_actions = self.get_actions(
                eval_obs, available_actions=eval_available_actions, add_random=False, 
                share_obs=eval_share_obs
            )
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if "smac" in self.args["env"]:
                        if eval_infos[eval_i][0]["won"]:
                            eval_battles_won += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0


            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)
                if "smac" in self.args["env"]:
                    print(
                        "Eval win rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
                            eval_battles_won / eval_episode, eval_avg_rew, eval_avg_len
                        )
                    )
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    eval_battles_won / eval_episode,
                                ],
                            )
                        )
                        + "\n"
                    )
                self.log_file.flush()
                self.writter.add_scalar(
                    "eval_average_episode_rewards", eval_avg_rew, step
                )

                self.writter.add_scalar(
                    "eval_average_episode_length", eval_avg_len, step
                )
                
                break
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    # Init messages if done
                    self.init_messages()    
        # self.eval_envs.close()

    @torch.no_grad()
    def get_actions(self, obs, available_actions=None, add_random=True, share_obs=None):
        """Get actions for rollout.
        Args:
            obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
            add_random: (bool) whether to add randomness
        Returns:
            actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action, msg, _, self.beta = self.actor[agent_id].get_actions(
                        obs[:, agent_id],
                        available_actions[:, agent_id],
                        add_random,
                        msgs=self.messages,
                        agent_id=agent_id
                    )
            actions.append(
                _t2n(
                    action
                ))
            
            # Restore Messages
            self.messages[:, agent_id] = msg
        return np.array(actions).transpose(1, 0, 2)

    def warmup(self):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        # obs: (n_threads, n_agents, dim)
        # share_obs: (n_threads, n_agents, dim)
        # available_actions: (threads, n_agents, dim)
        obs, share_obs, available_actions = self.envs.reset()
        self.messages = torch.zeros(obs.shape[0], obs.shape[1],
                                     available_actions.shape[-1]).to(self.device)
        
        for _ in range(warmup_steps):
            # action: (n_threads, n_agents, dim)
            actions = self.sample_actions(available_actions)
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(actions)
            
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2),
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2),
                actions 
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
        return obs, share_obs, available_actions
    
    def insert(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            messages
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        # valid_transition denotes whether each transition is valid or not (invalid if corresponding agent is dead)
        # shape: (n_threads, n_agents, 1)
        valid_transitions = 1 - self.agent_deaths

        self.agent_deaths = np.expand_dims(dones, axis=-1)

        # terms use False to denote truncation and True to denote termination
       
        terms = np.full(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            False,
        )
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            for agent_id in range(self.num_agents):
                if dones[i][agent_id]:
                    if not (
                        "bad_transition" in infos[i][agent_id].keys()
                        and infos[i][agent_id]["bad_transition"] == True
                    ):
                        terms[i][agent_id][0] = True

        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[i])
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros(
                    (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"].copy()
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"].copy()

       
        data = (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            np.expand_dims(dones, axis=-1),  # (n_threads, n_agents, 1)
            valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
            terms,  # (n_threads, n_agents, 1)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            messages
        )

        self.buffer.insert(data)


    def sample_actions(self, available_actions=None):
        """Sample random actions for warmup.
        Args:
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
        Returns:
            actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            for thread in range(self.algo_args["train"]["n_rollout_threads"]):
                if available_actions[thread] is None:
                    action.append(self.action_spaces[agent_id].sample())
                else:
                    action.append(
                        Categorical(
                            torch.tensor(available_actions[thread, agent_id, :])
                        ).sample()
                    )
            actions.append(action)
        if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
            return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)

        return np.array(actions).transpose(1, 0, 2)
