# HIGHLIGHTS

An implementation of the HIGHLIGHTS algorithm for Agent Policy Summarization: 
#### Amir, Ofra, Finale Doshi-Velez, and David Sarne. "Summarizing agent strategies." Autonomous Agents and Multi-Agent Systems 33.5 (2019): 628-644.

## Description:
Strategy summarization techniques convey agent behavior by demonstrating the actions taken by the agent in a selected set of world states. The key question in this approach is then how to recognize meaningful agent situations.

The HIGHLIGHTS algorithm extracts *important* states from execution traces of the agent based on some importance metric.
Intuitively, a state is considered important if the decision made in that state has a substantial impact on the agent's utility.

## Requirements:

The HIGHLIGHTS algorithm receives as input a **trained** agent and it's simulation environment.

The agent must be Q-value based and have access to these values given an observation or state. 

## Setup:
Install requirements:

`pip install -r requirements.txt`

make sure that: `gym <= 0.19`

Any trained agents should be in the *Agents* folder

## Utilizing Your Agent
To run HIGHLIGHTS on your agent the follwing files must be updated:

**get_agent.py**: In this file you must implement loading both your agent and its environment in the 'get_agent()' function.

**get_traces.py**: In this file you must implement running a single trace while using the Trace and State classes.


## Running
To run the HIGHLIGHTS algorithm: `python run.py`

### Parameter tuning
All hyper parameters can be observed and adjusted in in `run.py`.

#### Important Parameters:

1.`load_dir` - the path from which to load the agent traces `Traces.pkl` & `States.pkl`. If this is `None` then the algorithm will obtain execution traces which can later be used to skip this part.

2.`num_highlights` - the number of highlight videos to output

3.`trajectory_length` - the length of each video

# Example Implementation
We provide a working implementation of a ddqn agent in the highway domain.


## Dependencies:

The example implementation for the highway domain requires the following repositories:

[highway-env - v1.5](https://github.com/eleurent/highway-env/tree/v1.5), [rl-agents](https://github.com/eleurent/rl-agents)






