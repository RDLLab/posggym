"""Default config to use for training."""
from ray.rllib.algorithms.ppo.ppo import PPOConfig


def get_default_ppo_training_config(env_id: str, seed: int, log_level: str):
    """Get the Rllib agent config for an agent being trained."""
    config = PPOConfig()
    config.environment(
        env=env_id,
        env_config={
            "env_id": env_id,
            "flatten_obs": True,
        },
    )
    # Resource notes
    # trainer (or learner in version >2.3) worker = worker used for training NN
    # worker (without trainer/learner) = any other worker (e.g. rollout workers)
    config.resources(
        # Total number of gpus to allocate to the algorithm process. Can be a fraction.
        # For policy evaluation this would typically be set to 0, unless the NN are big
        num_gpus=1.0,
        # Number of cpus to allocate per worker
        num_cpus_per_worker=1,
        # Number of gpus to allocate per rollout worker. This can be a fraction.
        # Typically this should be 0 since we want rollout workers using CPU since they
        # don't do batch inference. Should only be > 0 if env itself required a GPU.
        num_gpus_per_worker=0.0,
        # Number of workers used for training. A value of 0 means training will take
        # place on a local worker on head node CPUs or 1 GPU (determined by
        # `num_gpus_per_learner_worker`).
        num_trainer_workers=0,
        # Number of CPUs allocated per trainer worker. Only necessary for custom
        # processing pipeline inside each Learner requiring multiple CPU cores.
        # Ignored if `num_learner_workers = 0`.
        num_cpus_per_trainer_worker=0,
        # Number of GPUs allocated per worker. If `num_learner_workers = 0`, any value
        # greater than 0 will run the training on a single GPU on the head node, while
        # a value of 0 will run the training on head node CPU cores.
        # This value should be num_gpus
        num_gpus_per_trainer_worker=1.0,
    )
    config.rollouts(
        # value of 1 uses a separate rollout worker to the main update worker
        num_rollout_workers=1,
        # Number of environments to evaluate vector-wise per worker. This enables model
        # inference batching, which can improve performance for inference bottlenecked
        # workloads.
        num_envs_per_worker=1,
        rollout_fragment_length="auto",
        observation_filter="NoFilter",
        batch_mode="truncate_episodes",
    )
    config.training(
        gamma=0.999,
        lr=0.0003,
        train_batch_size=2048,
        model={
            # === Model Config ===
            # === Built-in options ===
            # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
            # These are used if no custom model is specified and the input space is
            # 1D. Number of hidden layers to be used.
            "fcnet_hiddens": [64, 32],
            # Activation function descriptor.
            # Supported values are: "tanh", "relu", "swish" (or "silu"),
            # "linear" (or None).
            "fcnet_activation": "tanh",
            # Whether layers should be shared for the value function.
            "vf_share_layers": False,
            # == LSTM ==
            # Whether to wrap the model with an LSTM.
            "use_lstm": True,
            # Max seq len for training the LSTM, defaults to 20.
            "max_seq_len": 20,
            # Size of the LSTM cell.
            "lstm_cell_size": 256,
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            "lstm_use_prev_action": False,
            # Whether to feed r_{t-1} to LSTM.
            "lstm_use_prev_reward": False,
        },
        optimizer={},
        lr_schedule=None,
        use_critic=True,
        use_gae=True,
        lambda_=0.95,
        kl_coeff=0.2,
        sgd_minibatch_size=512,
        num_sgd_iter=2,
        shuffle_sequences=True,
        vf_loss_coeff=1.0,
        entropy_coeff=0.001,
        entropy_coeff_schedule=None,
        clip_param=0.2,
        vf_clip_param=1.0,
        grad_clip=5.0,
        kl_target=0.01,
    )
    config.framework(framework="torch")
    config.exploration(
        explore=True,
        exploration_config={
            "type": "StochasticSampling",
            # add some random timesteps to get agents away from initial "safe" starting
            # positions
            "random_timesteps": 5,
        },
    )
    config.debugging(
        # default is "WARN". Options (in order of verbosity, most to least) are:
        # "DEBUG", "INFO", "WARN", "ERROR"
        log_level="WARN",
        seed=seed,
        # disable by default since it can lead to many zombie processes when creating
        # lots of Algorithm instances
        log_sys_usage=False,
    )
    config.reporting(metrics_num_episodes_for_smoothing=100)
    return config
