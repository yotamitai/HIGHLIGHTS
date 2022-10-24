from os.path import join

import gym
import json
from os import listdir

from highway_env.envs.highway_env import HighwayEnv
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory


def get_agent(args):

    #---------- Implement here for specific agent and environment loading scheme ----------

    config_filename = [x for x in listdir(args.agent_path) if "metadata" in x][0]
    f = open(join(args.agent_path, config_filename))
    config = json.load(f)
    env_config, agent_config = config['env'], config['agent']
    env = gym.make(env_config["env_id"])
    env.seed(args.seed)
    env.configure(env_config)
    env.define_spaces()
    agent = agent_factory(env, agent_config)
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)

    # ---------- ----------

    return env, agent
