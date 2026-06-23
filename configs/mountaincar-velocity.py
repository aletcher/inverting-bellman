"""MountainCar experiment config.

Usage:
    python run.py --config configs/mountaincar-velocity.py
    python run.py --config configs/mountaincar-velocity.py --phases pqn,plot_pqn
    python run.py --config configs/mountaincar-velocity.py --LR 5e-4 --GAMMA 0.95
"""

import jax.numpy as jnp

# ── Env ───────────────────────────────────────────────────────────────────────

ENV_NAME = "MountainCar"
STATE_DIM = 2
ACTION_DIM = 3
STATE_LABELS = ["position", "velocity"]
STATE_RANGES = [(-1.2, 0.6), (-0.07, 0.07)]
ACTION_NAMES = ["left", "none", "right"]
EVAL_DIMS_2D = [(0, 1)]
MAX_STEPS_IN_EPISODE = 200

# ── Reward / goals ────────────────────────────────────────────────────────────

REWARD_TYPE = "sparse"
SIGMA = 0.1
A = 0.01
TERMINATE_ON_GOAL = True
NUM_GOALS = 4
GOALS = jnp.column_stack([jnp.zeros(NUM_GOALS), jnp.linspace(-0.07, 0.07, NUM_GOALS)])
GOAL_MASKS = jnp.tile(jnp.array([0.0, 1.0]), (NUM_GOALS, 1))
GOAL_INPUT_DIMS = [1]


# ── PQN ───────────────────────────────────────────────────────────────────────

GAMMA = 0.99
LR = 1e-4
NUM_ENVS = 256
NUM_STEPS = 64
MINIBATCH_SIZE = 256
TOTAL_TIMESTEPS = int(5e6)
DECAY_TIMESTEPS = int(5e6)
NUM_EPOCHS = 1

EPS_START = 1.0
EPS_FINISH = 0.1
EPS_DECAY = 0.5

USE_HER = True
HER_K = 4
HER_STRATEGY = "random"

EVAL_NUMBER = 20
EVAL_NUM_ENVS = 16
USE_WANDB = False

PQN_CONFIG = {
    "ENV_NAME": ENV_NAME,
    "REWARD_TYPE": REWARD_TYPE,
    "REWARD_SIGMA": SIGMA,
    "REWARD_A": A,
    "TERMINATE_ON_GOAL": TERMINATE_ON_GOAL,
    "MAX_STEPS_IN_EPISODE": MAX_STEPS_IN_EPISODE,
    "NUM_GOALS": NUM_GOALS,
    "STATE_DIM": STATE_DIM,
    "ACTION_DIM": ACTION_DIM,
    "STATE_RANGES": STATE_RANGES,
    "GOALS": GOALS,
    "REWARD_MASK": GOAL_MASKS,
    "GOAL_INPUT_DIMS": GOAL_INPUT_DIMS,

    "GAMMA": GAMMA,
    "LR": LR,
    "NUM_ENVS": NUM_ENVS,
    "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
    "DECAY_TIMESTEPS": DECAY_TIMESTEPS,
    "NUM_STEPS": NUM_STEPS,
    "MINIBATCH_SIZE": MINIBATCH_SIZE,
    "NUM_EPOCHS": NUM_EPOCHS,
    "LR_SCHEDULE": "linear",
    "LR_END": 1e-20,
    "MAX_GRAD_NORM": 10.0,

    "EPS_START": EPS_START,
    "EPS_FINISH": EPS_FINISH,
    "EPS_DECAY": EPS_DECAY,
    "USE_OPTIMISTIC_RESETS": True,
    "OPTIMISTIC_RESET_RATIO": 16,

    "USE_HER": USE_HER,
    "HER_K": HER_K,
    "HER_STRATEGY": HER_STRATEGY,

    "NETWORK_DENSE_HIDDEN_SIZE": 1024,
    "NETWORK_DENSE_LAYERS": 4,
    "NORM_TYPE": "layer_norm",
    "NETWORK_SIGMOID_OUTPUTS": True,

    "EVAL_NUMBER": EVAL_NUMBER,
    "EVAL_NUM_ENVS": EVAL_NUM_ENVS,
    "USE_WANDB": USE_WANDB,
    "SEED": 42,
    "WANDB_PROJECT": "inverting-bellman-mountaincar",
    "WANDB_ENTITY": None,
    "WANDB_LOG_INTERVAL": 1,
}


# ── World model ───────────────────────────────────────────────────────────────

WM_LR = 1e-4
WM_BATCH_SIZE = 4096
WM_DENSE_HIDDEN_SIZE = 256
WM_DENSE_LAYERS = 2
WM_ACTIVATION = "tanh"
SAMPLE_FROM_RESET = False  # sample states from env.reset() instead of uniform


WM_CONFIG = {
    "BATCH_SIZE": WM_BATCH_SIZE,
    "NUM_STEPS": 20_000,
    "LR": WM_LR,
    "LR_SCHEDULE": "cosine",
    "SEED": 0,
    "ACTIVATION": WM_ACTIVATION,
    "DENSE_HIDDEN_SIZE": WM_DENSE_HIDDEN_SIZE,
    "DENSE_LAYERS": WM_DENSE_LAYERS,
    "WM_LOSS": "l1",  # "l1" or "mse"
    "SAMPLE_FROM_RESET": SAMPLE_FROM_RESET,
    "RESIDUAL_PREDICTION": True,  # predict s + delta(s, a) instead of s' = f(s, a)
    "OUTPUT_INIT_SCALE": 0.01,  # small final-layer init so initial delta ≈ 0 (network starts at identity)
    "EVAL_HEATMAP_RES": 500,
}


# ── Value iteration (2D only) ─────────────────────────────────────────────────

VI_GRID_RES = 200
VI_MAX_ITER = 10000
VI_CONVERGENCE_THRESHOLD = 1e-10

# ── Env / eval metadata ───────────────────────────────────────────────────────


UNSEEN_GOALS = [
    # (i) Fast car: reach velocity v = 0.06.
    {
        "label": "Fast car",
        "goal": [0.0, 0.06], "mask": [0.0, 1.0],
        "reward_type": "sparse", "a": 0.01, "sigma": 0.1,
        "terminate_on_goal": True,
    },
    # (ii) Gentle car: reach top of hill x = 0.6 without exceeding |v| = 0.05
    # (the unconstrained optimal uses |v| >= 0.06).
    {
        "label": "Gentle car",
        "goal": [0.6, 0.0], "mask": [1.0, 0.0],
        "reward_type": "sparse", "a": 0.1, "sigma": 0.1,
        "terminate_on_goal": True,
        "forbidden": [
            {"dim": "velocity", "condition": "lt", "threshold": -0.05},
            {"dim": "velocity", "condition": "gt", "threshold": 0.05},
        ],
    },
    # (iii) Shortest path: reach one of two goals x = -1.0 or x = 0.4.
    {
        "label": "Shortest path",
        "goals": [[-1.0, 0.0], [0.4, 0.0]], "mask": [1.0, 0.0],
        "reward_type": "sparse", "a": 0.1, "sigma": 0.1,
        "terminate_on_goal": True,
    },
]

ENV_CONFIG = {
    "ENV_NAME": ENV_NAME,
    "STATE_DIM": STATE_DIM,
    "STATE_LABELS": STATE_LABELS,
    "STATE_RANGES": STATE_RANGES,
    "ACTION_DIM": ACTION_DIM,
    "ACTION_NAMES": ACTION_NAMES,
    "EVAL_DIMS_2D": EVAL_DIMS_2D,
    "VI_GRID_RES": VI_GRID_RES,
    "VI_MAX_ITER": VI_MAX_ITER,
    "VI_CONVERGENCE_THRESHOLD": VI_CONVERGENCE_THRESHOLD,
    "UNSEEN_GOALS": UNSEEN_GOALS,
    "UNSEEN_NUM_EVAL_STARTS": 512,
}
