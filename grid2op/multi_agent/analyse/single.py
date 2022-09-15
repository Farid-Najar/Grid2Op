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


def _run_simple_actor(
    env : BaseEnv,
    actor : BaseAgent,
    nb_episodes : int,
    save_path : str
) -> dict:
    
    T = np.zeros(nb_episodes, dtype = int)
    obs = env.reset()
    t = 0
    
    print("Running simple simulation !")
    from tqdm import tqdm
    
    rewards_history = [[] for _ in range(nb_episodes)]
    cumulative_reward = np.zeros(nb_episodes)

    reward = 0
    
    for episode in tqdm(range(nb_episodes)):
        
        while True:
            t += 1
            
            action = actor.act(observation = obs, reward = reward)
            obs, reward, done, info = env.step(action)
            
            #obs._obs_env = None
            rewards_history[episode].append(reward)

            if done:
                #mean_rewards_history[episode] = np.mean(rewards_history)
                #std_rewards_history[episode] = np.std(rewards_history)
                cumulative_reward[episode] = np.sum(rewards_history[episode])
                obs = env.reset()
                T[episode] = t
                t = 0
                
                break
        if (episode+1)%10==0:
            np.save(save_path+'/'+f'single_cum_rewards{episode}.npy', arr=cumulative_reward)
            np.save(save_path+'/'+f'single_T{episode}.npy', arr=T)
            
    return {
        'rewards' : rewards_history,
        #'mean_rewards' : mean_rewards_history,
        #'std_rewards' : std_rewards_history,
        'episode_len' : T,
        'cumulative_reward' : cumulative_reward
    }

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
    mean_rewards_history = np.zeros(nb_episodes)
    std_rewards_history = np.zeros(nb_episodes)
    cumulative_reward = np.zeros(nb_episodes)
    
    info_history = [[] for _ in range(nb_episodes)]
    
    local_actions = [[] for _ in range(nb_episodes)]
    
    done_history = [[] for _ in range(nb_episodes)]
    
    actions_history = [[] for _ in range(nb_episodes)]
    
    obs_history = [[] for _ in range(nb_episodes)]
    
    r = 0
    
    for episode in tqdm(range(nb_episodes)):
        while True:
            t += 1
            
            obs_history[episode].append(obs[ma_env.agents[0]].to_vect())
            
            actions = {
                agent : actors[agent].act(observation = obs[agent], reward = r)
                for agent in ma_env.agents
            }
            obs, reward, dones, info = ma_env.step(actions)

            r = reward[ma_env.agents[0]]
            rewards_history[episode].append(r)
            info_history[episode].append(info[ma_env.agents[0]].copy())
            
            #for agent in ma_env.agents:
            #    # TODO pourquoi ce problème ?
            #    obs[agent]._obs_env = None
                
            local_actions[episode].append(actions.copy())
            done_history[episode].append(dones[ma_env.agents[0]])
            actions_history[episode].append(ma_env.global_action.copy())
                

            if dones[ma_env.agents[0]]:
                #mean_rewards_history[episode] = np.mean(rewards_history)
                #std_rewards_history[episode] =  np.std(rewards_history)
                cumulative_reward[episode] = np.sum(rewards_history[episode])
                
                obs = ma_env.reset()
                T[episode] = t
                t = 0
                break
        if (episode+1)%10 == 0 and episode>0:
            res = {
                'rewards' : rewards_history,
                'observations' : obs_history,
                'episode_len' : T,
                'info_history' : info_history,
                'local_actions' : local_actions,
                'done_history' : done_history,
                'actions' : actions_history,
                'cumulative_reward' : cumulative_reward
            }
            np.save(save_path+'/'+f'res_ma{episode}.npy', arr=res)
            
            
    return {
        'rewards' : rewards_history,
        #'mean_rewards' : mean_rewards_history,
        #'std_rewards' : std_rewards_history,
        'observations' : obs_history,
        'episode_len' : T,
        'info_history' : info_history,
        'local_actions' : local_actions,
        'done_history' : done_history,
        'actions' : actions_history,
        'cumulative_reward' : cumulative_reward
    }

    
def compare_simple_and_multi(
    ma_env, # It is grid2op.multi_agent.multiAgentEnv.MultiAgentEnv
    simple_actor : BaseAgent, 
    ma_actors : MAAgents, 
    episodes : int = 2,
    seed = 0,
    chronics_id = 0,
    save_path = "./",
    ):
    
    ma_env.seed(seed)
    ma_env._cent_env.set_id(chronics_id)
    
    results_simple = _run_simple_actor(ma_env._cent_env, simple_actor, episodes, save_path)
    np.save(save_path+'/'+f'results_simple{episodes}.npy', arr=results_simple)
    
    
    ma_env.seed(seed)
    ma_env._cent_env.set_id(chronics_id)
    results_ma = _run_ma_actors(ma_env, ma_actors, episodes, save_path)
    np.save(save_path+'/'+f'results_ma{episodes}.npy', arr=results_ma)
    
    #save results
    # TODO
    
    return results_simple, results_ma


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
            
from grid2op.Agent import TopologyGreedy

ma_env.seed(0)
ma_env._cent_env.set_id(0)
simple_actor = TopologyGreedy(ma_env._cent_env.action_space)

results_simple1 = _run_simple_actor(ma_env._cent_env, simple_actor, 20, save_path='./res_single20')