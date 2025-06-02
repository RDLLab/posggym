import timeit
from functools import partial

import numpy as np
import torch
from posggym.envs.continuous.predator_prey_continuous import PredatorPreyContinuousEnv
from posggym.envs.differentiable.predator_prey_diff import PredatorPreyDiff
from posggym.vector.sync_vector_env import SyncVectorEnv


if __name__ == "__main__":
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:

        def env_fn():
            return PredatorPreyContinuousEnv(
                world="20x20Blocks", num_predators=7, num_prey=7
            )

        envs = SyncVectorEnv([env_fn for _ in range(batch_size)])

        actions = {
            i: np.stack([act_space.sample() for _ in range(batch_size)])
            for i, act_space in envs.single_action_spaces.items()
        }

        def step(envs_, actions_):
            envs_.step(actions_)

        # Create a partial function with envs and actions as bound arguments
        step_with_args = partial(step, envs_=envs, actions_=actions)

        # Time the step function over 20 executions
        execution_time = timeit.timeit(step_with_args, number=20)

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        env = PredatorPreyDiff(
            batch_size=batch_size,
            world="20x20Blocks",
            num_predators=7,
            num_prey=7,
        )

        def batch_sample_(a_s, batch_size):
            return np.array([a_s.sample() for _ in range(batch_size)])

        # Take a random action as input to the step function
        a = {
            i: torch.Tensor(batch_sample_(env.action_spaces[i], batch_size))
            for i in env.agents
        }
        for action in a.values():
            action.requires_grad_(True)

        # Function to benchmark the step function
        def benchmark_step1(env, a):
            env.step(a)

        benchmark_step_with_args = partial(benchmark_step1, env=env, a=a)

        # Time the step function over 20 executions
        execution_time = timeit.timeit(benchmark_step_with_args, number=20)
        print(f"Average time per step: {execution_time / 20} seconds")

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        env = PredatorPreyDiff(
            batch_size=batch_size,
            world="20x20Blocks",
            num_predators=7,
            num_prey=7,
        )

        def batch_samples(a_s, batch_size):
            return np.array([a_s.sample() for _ in range(batch_size)])

        # Take a random action as input to the step function
        a = {
            i: torch.Tensor(batch_samples(env.action_spaces[i], batch_size))
            for i in env.agents
        }

        # Function to benchmark the step function
        def benchmark_step2(env, a):
            with torch.no_grad():
                env.step(a)

        benchmark_step_with_args = partial(benchmark_step2, env=env, a=a)

        # Time the step function over 20 executions
        execution_time = timeit.timeit(benchmark_step2, number=20)
        print(f"Average time per step: {execution_time / 20} seconds")
