from grid2op import make
from grid2op.Action.PlayableAction import PlayableAction
from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv
import numpy as np
    

from lightsim2grid import LightSimBackend
bk_cls = LightSimBackend

action_domains = {
    'agent_0' : [0,1,2,3, 4],
    'agent_1' : [5,6,7,8,9,10,11,12,13]
}

from gym.spaces import Discrete, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv as MAEnv

class MAEnvWrapper(MAEnv):
    def __init__(self, env_config=None):
        super().__init__
        
        env_name = "l2rpn_case14_sandbox"#"educ_case14_storage"
        env = make(env_name, test=False, backend = bk_cls(),
                action_class=PlayableAction, _add_to_name="_test_ma")


        self.ma_env = MultiAgentEnv(env, action_domains, copy_env=False)
        self._agent_ids = set(self.ma_env.agents)
        self.ma_env.seed(42)
        obs = self.ma_env.reset()[self.ma_env.agents[0]].to_vect()
        self.observation_space = Box(shape=obs.shape, high=np.infty, low=-np.infty)#{
        #    agent : Box(shape=obs.shape, high=np.infty, low=-np.infty)
        #    for agent in self.ma_env.agents
        #}
        
        self.all_actions = {
            agent : self._get_tested_action(self.ma_env.action_spaces[agent])
            for agent in self.ma_env.agents
        }
        self.action_space = {
            agent : Discrete(len(self.all_actions[agent]))
            for agent in self.ma_env.agents
        }
        
    def reset(self):
        obs = self.ma_env.reset()
        o = obs[self.ma_env.agents[0]].to_vect()
        return {
            agent : o.copy()
            for agent in self.ma_env.agents
        }
    
    def step(self, actions):
        #print(np.array(actions['agent_0'], dtype=int).shape)
        #print({
        #    agent : np.argmax(actions[agent])
        #    for agent in self.ma_env.agents
        #})
        a = {
            agent : self.all_actions[agent][np.argmax(actions[agent])]
            for agent in self.ma_env.agents
        }
        
        obs, r, info, done = self.ma_env.step(a)
        done['__all__'] = done[self.ma_env.agents[0]]
        #info['__all__'] = ''
        o = obs[self.ma_env.agents[0]].to_vect()
        obs = {
            agent : o.copy()
            for agent in self.ma_env.agents
        }
        info = dict()
        return obs, r, done, info
    
    def _get_tested_action(self, action_space):
        res = [action_space({})]  # add the do nothing
        # better use "get_all_unitary_topologies_set" and not "get_all_unitary_topologies_change"
        # maybe "change" are still "bugged" (in the sens they don't count all topologies exactly once)
        res += action_space.get_all_unitary_topologies_set(action_space)
        return res
    
import ray
from ray.rllib.agents.maddpg import MADDPGTrainer

import pandas as pd
import json
import os
import shutil
import sys

checkpoint_root = "./maddpg_test"
# Where checkpoints are written:
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)

# Where some data will be written and used by Tensorboard below:
ray_results = f'{os.getenv("HOME")}/ray_results/'
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

info = ray.init(ignore_reinit_error=True)
print("Dashboard URL: http://{}".format(info["webui_url"]))

new_ma_env = MAEnvWrapper()

#Configs
SELECT_ENV = MAEnvWrapper                            # Specifies the OpenAI Gym environment for Cart Pole
N_ITER = 1000                                     # Number of training runs.

config = MADDPGTrainer.get_default_config().copy()              # PPO's default configuration. See the next code cell.
config["log_level"] = "WARN"                    # Suppress too many messages, but try "INFO" to see what can be printed.

# Other settings we might adjust:
config["num_workers"] = 9                       # Use > 1 for using more CPU cores, including over a cluster
#config["num_sgd_iter"] = 10                     # Number of SGD (stochastic gradient descent) iterations per training minibatch.
                                                # I.e., for each minibatch of data, do this many passes over it to train. 
#config["sgd_minibatch_size"] = 64              # The amount of data records per minibatch
config["actor_hiddens"] = [200, 100, 100]
config["critic_hiddens"] = [200, 100, 100]
config["critic_lr"] = 5e-3
config["actor_lr"] =  5e-3
#config["model"]["fcnet_hiddens"] = [100, 50]    #
#config["num_cpus_per_worker"] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed
#config["vf_clip_param"] = 100

from ray.rllib.policy.policy import PolicySpec, Policy
DO_NOTHING_EPISODES = 200

class DoNothing(Policy):
    def __init__(self, observation_space, action_space, config):
        
        super().__init__(observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        *args,
                        **kwargs):
        """Compute actions on a batch of observations."""
        return [0 for _ in obs_batch], [], {}
    
    def learn_on_batch(self, samples):
        """No learning."""
        #return {}
        pass
    def get_weights(self):
        pass
    def set_weights(self, weights):
        pass
    
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == 'agent_1':
        return agent_id

    return "do_nothing" if episode.episode_id < DO_NOTHING_EPISODES else "agent_0"
    

config["multiagent"] = {
    "policies" : {
        "agent_0" : PolicySpec(
            action_space=Discrete(len(new_ma_env.all_actions["agent_0"])),
            config={
                "agent_id" : 0
            }
        ),
        "agent_1" : PolicySpec(
            action_space=Discrete(len(new_ma_env.all_actions["agent_1"])),
            config={
                "agent_id" : 1
            }
        ),
        #"do_nothing" : PolicySpec(
        #    action_space=Discrete(1),
        #    policy_class=DoNothing)
    },
    "policy_mapping_fn":
            lambda x, *args, **kwargs : x#policy_mapping_fn,
    #"policies_to_train": ["agent_0", "agent_1"],
}
config["framework"] = "tf"
config["eager_tracing"] = False

#Trainer
agent = MADDPGTrainer(config, env=SELECT_ENV)

results = []
episode_data = []
episode_json = []

for n in range(N_ITER):
    result = agent.train()
    results.append(result)
    
    episode = {'n': n, 
               'episode_reward_min': result['episode_reward_min'], 
               'episode_reward_mean': result['episode_reward_mean'], 
               'episode_reward_max': result['episode_reward_max'],  
               'episode_len_mean': result['episode_len_mean']}
    
    episode_data.append(episode)
    episode_json.append(json.dumps(episode))
    file_name = agent.save(checkpoint_root)
    
    print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}. Checkpoint saved to {file_name}')

    with open('maddpg_test/rewards.json', 'w') as outfile:
        json.dump(episode_json, outfile)