# POSGGym

The goal of this library is to provide implementations of Partially Observable Stochastic Game (POSG) environments coupled with dynamic models of each environment, all under a unified API. While there are a number of amazing open-source implementations for POSG environments, very few have easily useable dynamic models that can be used for planning. The aim of this library is to fill this gap.

Another aim it to provide open-source implementations for many of the environments commonly used in the Partially-Observable multi-agent planning literature. While some open-source implementations exist for some of the common environments, we hope to provide a central repository, with easy to understand and use implementations in order to make reproducibility easier and to aid in faster research.


## Installation

At the moment we only support installation by cloning the repo and installing locally.

Once the repo is cloned, you can install POSGGym using PIP by navigating to the `posggym` root directory (the one containing the `setup.py` file), and running:

```
pip install -e .
```

Or use the following to install `posggym` with all dependencies:

```
pip install -e .[all]
```


## Environment API (OUTDATED)

POSGGym environments follow the [Open AI Gym](https://github.com/openai/gym) API, except using joint actions, observations and rewards.

Creating environment instances and interacting with them is very simple, and flows almost identically to the Open AI Gym user flow. Here's an example using the "TwoPaths-v0" environment:

```
import posggym

env = posggym.make("TwoPaths-v0")

# env is created now we can use it

# the number of agents in the environment is stored in the n_agents property
# env.n_agents
for episode in range(10):

	# reset the environment and get initial observation
	# observations is a tuple with one observation per agent
	observations = env.reset()

	for step in range(50):
		# select an action for each agent in the environment
		# actions are a tuple of agent actions, ordered by agent index
		actions = (as_i.sample() for as_i in env.action_spaces)

		# execute a step
		# observations and rewards are tuples with one entry per agent
		# done signifies if episode reached a terminal state
		# info is auxiliary information, e.g. outcome after the step
		observations, rewards, done, info = env.step(actions)

		# the env can also be rendered
		# where "human" can be replaced by different rendering modes
		env.render("human")
```


## Model API (OUTDATED)

Each environment comes with an implemented POSG model. Every environment model implements a generative model, which can be used for planning, along with an initial belief. Some environments also implement the full POSG model including the transition, joint observation and joint reward functions.

The following is an example of the generative model API.


```
import posggym

env = posggym.make("TwoPaths-v0")
model = env.model

initial_belief = model.initial_belief

# could also use model.sample_initial_state() convenience function
state = initial_belief.sample()

observations = model.sample_initial_obs(state)

# get actions to perform
# e.g. policy(observation[i]) for each agent i
actions = (as_i.sample() for as_i in env.action_spaces)

# next_state, observations, rewards, done, outcomes ~ model.step(state, actions)
joint_step = model.step(state, actions)

state = joint_step.state
observations = joint_step.observations
rewards = joint_step.rewards
done = joint_step.done           # whether terminal state was reached
outcomes = joint_step.outcomes   # e.g. winner and loser agents if applicable
```


## Model Types

POSGGym supports two types of models:

- *Full* (`POSGFullModel` Class) - models which are fully defined, in the sense they include implementations of every component of the POSG model (including state space, transition, observation, and reward functions)
- *Generative* (`POSGModel` Class) - models which only include implementation of a generative function which takes in a state and joint action and returns the next state, joint observation, joint reward, and terminal information.

*Full* models implement a superset of the generative model functionallity.

## Implemented Environments

POSGGym includes the following families of environments along with some third-party environments. The code for implemented environments are located in the `posggym/envs/` subdirectory.

- *Classic* - These are classic POSG problems.
- *Grid-World* - These environments are all based in a 2D Gridworld.


## Authors

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


## License

`MIT` © 2022, Jonathon Schwartz


## Versioning

The POSGGym library uses [semantic versioning](https://semver.org/).
