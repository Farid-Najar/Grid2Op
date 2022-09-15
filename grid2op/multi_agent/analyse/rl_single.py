from grid2op.Agent import BaseAgent
from grid2op.Converter.IdToAct import IdToAct



from grid2op.multi_agent.ma_typing import LocalObservation, LocalObservationSpace, \
    LocalAction, LocalActionSpace 

from grid2op import make
from grid2op.Action.PlayableAction import PlayableAction
from grid2op.Action import BaseAction
from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv
import numpy as np
from grid2op.multi_agent.multi_agentExceptions import *

    
import numpy as np

from lightsim2grid import LightSimBackend
bk_cls = LightSimBackend

action_domains = {
    'agent_0' : [0,1,2,3, 4],
    'agent_1' : [5,6,7,8,9,10,11,12,13]
}

import gym
from gym.spaces import Discrete, Box

class EnvWrapper(gym.Env):
    def __init__(self, env_config=None):
        env_name = "l2rpn_case14_sandbox"#"educ_case14_storage"
        self.env = make(env_name, test=False, backend = bk_cls(),
                        action_class=PlayableAction, _add_to_name="_test_ma")
        self.env.seed(0)
        self.all_actions = self._get_tested_action(self.env.action_space)
        obs = self.env.reset().to_vect()
        
        self.observation_space = Box(shape=obs.shape, high=np.infty, low=-np.infty)#GymObservationSpace(self.env)
        self.action_space = Discrete(len(self.all_actions))
        
        self.reward_range = self.env.reward_range
        
        
    def reset(self):
        obs = self.env.reset()
        return obs.to_vect()
    
    def step(self, action):
        a = self.all_actions[action]
        
        obs, r, info, done = self.env.step(a)
        
        return obs.to_vect(), r, info, done
    
    def _get_tested_action(self, action_space):
        res = [action_space({})]  # add the do nothing
        # better use "get_all_unitary_topologies_set" and not "get_all_unitary_topologies_change"
        # maybe "change" are still "bugged" (in the sens they don't count all topologies exactly once)
        res += action_space.get_all_unitary_topologies_set(action_space)
        return res
  
import ray
from ray.rllib.agents.ppo import ppo

import pandas as pd
import json
import os
import shutil
import sys

checkpoint_root = "./single_ppo2"
# Where checkpoints are written:
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)

# Where some data will be written and used by Tensorboard below:
ray_results = f'{os.getenv("HOME")}/ray_results/'
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

info = ray.init(ignore_reinit_error=True)
print("Dashboard URL: http://{}".format(info["webui_url"]))

SELECT_ENV = EnvWrapper                            # Specifies the OpenAI Gym environment for Cart Pole
N_ITER = 1000                                     # Number of training runs.

config = ppo.DEFAULT_CONFIG.copy()              # PPO's default configuration. See the next code cell.
config["log_level"] = "WARN"                    # Suppress too many messages, but try "INFO" to see what can be printed.

# Other settings we might adjust:
config["num_workers"] = 8                       # Use > 1 for using more CPU cores, including over a cluster
config["num_sgd_iter"] = 10                     # Number of SGD (stochastic gradient descent) iterations per training minibatch.
                                                # I.e., for each minibatch of data, do this many passes over it to train. 
config["sgd_minibatch_size"] = 64              # The amount of data records per minibatch
config["model"]["fcnet_hiddens"] = [256, 256]#[200, 100, 50]    #
config["num_cpus_per_worker"] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed
config["vf_clip_param"] = 100
#config['disable_env_checking']=True

agent = ppo.PPOTrainer(config, env=SELECT_ENV)

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
    
    with open('single_ppo2/rewards.json', 'w') as outfile:
        json.dump(episode_json, outfile)


        