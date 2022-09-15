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
env_name = "l2rpn_case14_sandbox"#"educ_case14_storage"
env = make(env_name, test=False, backend = bk_cls(),
                action_class=PlayableAction, _add_to_name="_test_ma")


ma_env = MultiAgentEnv(env, action_domains, copy_env=False)

ma_env.seed(0)
obs = ma_env.reset()
        

from grid2op.multi_agent.ma_typing import MAAgents
from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Agent.baseAgent import BaseAgent
from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv

def _run_ma_actors(
    ma_env : MultiAgentEnv,
    actors : MAAgents,
    nb_episodes : int,
    save_path : str
) -> dict:
    
    print("Running multi-agent simulation !")
    
    T = np.zeros(nb_episodes, dtype = int)
    obs = ma_env.reset()
    t = 0
    
    from tqdm.notebook import tqdm
    
    rewards_history = [[] for _ in range(nb_episodes)]
    cumulative_reward = np.zeros(nb_episodes)
    
    local_actions = []
    
    actions_history = []
    
    obs_history = []
    
    r = 0
    
    for episode in tqdm(range(nb_episodes)):
        while True:
            t += 1
            
            obs_history.append(obs[ma_env.agents[0]].to_vect())
            
            actions = {
                agent : actors[agent].act(observation = obs[agent], reward = r)
                for agent in ma_env.agents
            }
            obs, reward, dones, _ = ma_env.step(actions)

            r = reward[ma_env.agents[0]]
            rewards_history[episode].append(r)
            
            local_actions.append(actions.copy())
            actions_history.append(ma_env.global_action.copy())

            if dones[ma_env.agents[0]]:
                #mean_rewards_history[episode] = np.mean(rewards_history)
                #std_rewards_history[episode] =  np.std(rewards_history)
                cumulative_reward[episode] = np.sum(rewards_history[episode])
                np.save(save_path+'/'+f'observations{episode}.npy', arr=obs_history)
                obs_history = []
                np.save(save_path+'/'+f'local_actions{episode}.npy', arr=local_actions)
                local_actions = []
                np.save(save_path+'/'+f'actions{episode}.npy', arr=actions_history)
                actions_history = []
                obs = ma_env.reset()
                T[episode] = t
                t = 0
                break
        if (episode+1)%10 == 0 and episode>0:
            res = {
                'rewards' : rewards_history,
                'episode_len' : T,
                'cumulative_reward' : cumulative_reward
            }
            np.save(save_path+'/'+f'res_ma{episode}.npy', arr=res)
            


class Predictor:
    def __init__(self, 
                 action_space,
                 do_nothing = False,
                 model = None,
                 nn_kwargs = {}):
        
        self.action_space = action_space
        self.do_nothing = do_nothing
        self.model = model
        
        res = [self.action_space({})]  # add the do nothing
        res += self.action_space.get_all_unitary_topologies_set(self.action_space)
        self.all_actions = res
        
    
    def predict(self, observation : LocalObservation) -> LocalAction:
        if self.do_nothing:
            return self.action_space({})
        elif self.model is not None:
            #TODO return the prediction
            a = self.model.predict([observation.to_vect()])[0]
            return self.all_actions[a]
        else:
            raise("Model is missing !")
            
from typing import List, Optional
import copy

class LocalTopologyGreedyExpert(BaseAgent):
    def __init__(self,
                 agent_nm,
                 action_space,
                 ma_env,
                 predictors : Optional[List[Predictor]] = None,
                 **kwargs):
        super().__init__(action_space)
        
        self.agent_nm = agent_nm
        self.other_agents = [
            agent
            for agent in ma_env.agents if agent != agent_nm
        ]
        self.executed_actions = set()
        
        self.action_space = action_space
        
        self.curr_iter = 0
        self.ma_env = ma_env
        
        self.global_action_space = ma_env._cent_env.action_space
        
        if predictors is None:
            self.predictors = {
                agent : Predictor(ma_env.action_spaces[agent], do_nothing=True)
                for agent in self.other_agents
            }
        else:
            self.predictors = predictors
            
        self.tested_action = None
            
    def act(self, observation : LocalObservation, reward, done = False):
        self.curr_iter += 1

        # Look for overloads and rank them
        #ltc_list = self.getRankedOverloads(observation)
        #counterTestedOverloads = 0
        #overloaded = np.any(observation.rho >= 1)
        #
        #if not overloaded:
        #    return self.action_space({})
        #else:
        other_actions = {
            agent : self.predictors[agent].predict(observation)
            for agent in self.other_agents
        }
        
        self.tested_action = self._get_tested_action(observation)
        if len(self.tested_action) > 1:
            self.resulting_rewards = np.full(
                shape=len(self.tested_action), fill_value=-np.infty, dtype=float
            )
            for i, action in enumerate(self.tested_action):
                actions = copy.deepcopy(other_actions)
                actions.update({self.agent_nm : action})
                a = action.to_global(self.ma_env._cent_env.action_space)
                is_legal, reason = self.ma_env._cent_env._game_rules(action=a, env=self.ma_env._cent_env)
                if is_legal:
                    (
                        simul_obs,
                        simul_reward,
                        simul_has_error,
                        simul_info,
                    ) = observation.simulate(actions)
                    self.resulting_rewards[i] = simul_reward
                else:
                    self.resulting_rewards[i] = -np.infty

            reward_idx = int(
                np.argmax(self.resulting_rewards)
            )
            
            best_action = self.tested_action[reward_idx]
        else:
            best_action = self.tested_action[0]
            
        return best_action
        
    def _get_tested_action(self, observation):
        if self.tested_action is None:
            res = [self.action_space({})]  # add the do nothing
            # better use "get_all_unitary_topologies_set" and not "get_all_unitary_topologies_change"
            # maybe "change" are still "bugged" (in the sens they don't count all topologies exactly once)
            res += self.action_space.get_all_unitary_topologies_set(self.action_space)
            self.tested_action = res
        return self.tested_action
            
    def reset(self, observation):
        # No internal states to reset
        pass

    def load(self, path):
        # Nothing to load
        pass

    def save(self, path):
        # Nothing to save
        pass


ma_env.seed(0)
ma_env._cent_env.set_id(0)
episodes = 20
ma_actors = dict()
for agent_nm in ma_env.agents:
    ma_actors[agent_nm] = LocalTopologyGreedyExpert(
        agent_nm,
        ma_env.action_spaces[agent_nm],
        ma_env
    )

results_ma = _run_ma_actors(ma_env, ma_actors, episodes, save_path='./res_ma00_20')
