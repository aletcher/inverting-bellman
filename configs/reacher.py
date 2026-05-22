"""Reacher experiment config.

Usage:
    python run.py --config configs/reacher.py
    python run.py --config configs/reacher.py --phases pqn,plot_pqn
    python run.py --config configs/reacher.py --LR 5e-4 --GAMMA 0.95
"""

import jax.numpy as jnp

from envs.reacher_utils import make_action_table

# ── Env ───────────────────────────────────────────────────────────────────────

ENV_NAME = "Reacher"

# Actions: N×N grid per joint (N torque values uniformly spaced in [-1, 1]).
REACHER_N_TORQUES = 3
REACHER_TORQUE_VALUES = jnp.linspace(-1.0, 1.0, REACHER_N_TORQUES).tolist()
_action_table = make_action_table([REACHER_TORQUE_VALUES, REACHER_TORQUE_VALUES])

STATE_DIM = 6
ACTION_DIM = len(_action_table)
STATE_LABELS = ["t1", "t2", "w1", "w2", "fp_x", "fp_y"]
STATE_RANGES = [
    (-float(jnp.pi), float(jnp.pi)),
    (-float(jnp.pi), float(jnp.pi)),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
]
ACTION_NAMES = [f"({v1:.2g},{v2:.2g})" for v1, v2 in _action_table.tolist()]
EVAL_DIMS_2D = [(4, 5)]  # fingertip xy for trajectory plots
OBS_INPUT_DIMS = [0, 1, 2, 3]
MAX_STEPS_IN_EPISODE = 100

# ── Reward / goals ────────────────────────────────────────────────────────────

REWARD_TYPE = "sparse"
SIGMA = 0.1
A = 0.2
TERMINATE_ON_GOAL = True
# Polar sampling of fingertip targets within the reachable disk (radius 2).
_MAX_RADIUS = 1.0  # stay away from the singular boundary at r=2
_N_RADII = 1
_N_ANGLES = 4
_radii = jnp.linspace(_MAX_RADIUS / _N_RADII, _MAX_RADIUS, _N_RADII)
_angles = jnp.linspace(0, 2 * jnp.pi, _N_ANGLES, endpoint=False)
_R, _A = jnp.meshgrid(_radii, _angles, indexing="ij")
_ring_pts = jnp.stack([_R.ravel() * jnp.cos(_A.ravel()),
                       _R.ravel() * jnp.sin(_A.ravel())], axis=-1)
_grid = _ring_pts
NUM_GOALS = len(_grid)
GOALS = jnp.zeros((NUM_GOALS, STATE_DIM)).at[:, 4:6].set(_grid)
GOAL_MASKS = jnp.zeros((NUM_GOALS, STATE_DIM)).at[:, 4:6].set(1.0)
GOAL_INPUT_DIMS = [4, 5]  # only feed fingertip (x, y) goal dims to Q-network

# ── PQN ───────────────────────────────────────────────────────────────────────


REW_SCALE = 1.0
GAMMA = 0.99
LR = 1e-4
NUM_ENVS = 256
NUM_STEPS = 64
MINIBATCH_SIZE = 256
TOTAL_TIMESTEPS = int(1e7)
DECAY_TIMESTEPS = int(1e7)
NUM_EPOCHS = 1

EPS_START = 1.0
EPS_FINISH = 0.1
EPS_DECAY = 0.5

USE_HER = True
HER_K = 4
HER_STRATEGY = "future"

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

    "REW_SCALE": REW_SCALE,
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

    "NETWORK_DENSE_HIDDEN_SIZE": 2048,
    "NETWORK_DENSE_LAYERS": 6,
    "NORM_TYPE": "layer_norm",
    "NETWORK_SIGMOID_OUTPUTS": True,
    "OBS_INPUT_DIMS": OBS_INPUT_DIMS,

    "EVAL_NUMBER": EVAL_NUMBER,
    "EVAL_NUM_ENVS": EVAL_NUM_ENVS,
    "USE_WANDB": USE_WANDB,
    "SEED": 42,
    "WANDB_PROJECT": "inverting-bellman-reacher",
    "WANDB_ENTITY": None,
    "WANDB_LOG_INTERVAL": 10,
}

# ── World model ───────────────────────────────────────────────────────────────

WM_LR = 1e-4
WM_BATCH_SIZE = 1024
WM_DENSE_HIDDEN_SIZE = 256
WM_DENSE_LAYERS = 2
WM_ACTIVATION = "tanh"
SAMPLE_FROM_RESET = True

WM_CONFIG = {
    "BATCH_SIZE": WM_BATCH_SIZE,
    "NUM_STEPS": 1000,
    "LR": WM_LR,
    "LR_SCHEDULE": "linear",
    "SEED": 0,
    "ACTIVATION": WM_ACTIVATION,
    "DENSE_HIDDEN_SIZE": WM_DENSE_HIDDEN_SIZE,
    "DENSE_LAYERS": WM_DENSE_LAYERS,
    "WM_LOSS": "l1",  # "l1" or "mse"
    "ANGLE_DIMS": [0, 1],
    "WM_OUTPUT_DIM": 4,
    # WM net input is 4D effective state [θ₁, θ₂, ω₁, ω₂] + action one-hot.
    # The 4-vs-6 input choice is independent of WM_OUTPUT_DIM (which crops the
    # output) — set both to make the WM a true 4D→4D dynamics function.
    "WM_INPUT_DIMS": [0, 1, 2, 3],
    "SAMPLE_FROM_RESET": SAMPLE_FROM_RESET,
    "RESIDUAL_PREDICTION": True,
    "OUTPUT_INIT_SCALE": 0.01,
    "EVAL_HEATMAP_RES": 80,
}

# ── Value iteration — 4D effective state (theta1, theta2, omega1, omega2) ────
# Reacher's 6D raw obs is a deterministic function of the 4D underlying state,
# so VI on the 4D grid recovers Q* on the full obs manifold.

VI_GRID_RES = 50
VI_MAX_ITER = 2000
VI_STATE_DIM = 4
VI_STATE_RANGES = [
    (-float(jnp.pi), float(jnp.pi)),
    (-float(jnp.pi), float(jnp.pi)),
    (-2.0, 2.0),
    (-2.0, 2.0),
]
VI_STATE_LABELS = ["theta1", "theta2", "omega1", "omega2"]
# Headline Q_MSE / Q_MAPE eval support:
#   False (default, paper convention) — eval on the full VI grid (uniform on
#       VI_STATE_RANGES). Broader than the WM training support on ω: the WM
#       is trained on env.reset() (|ω|<=1) but Q is graded across |ω|<=4.
#   True — eval on the same env.reset() distribution the WM trains on, so
#       Q_MSE / WM_MSE share their support and the side-by-side is fair.
Q_EVAL_ON_TRAIN_SUPPORT = False


UNSEEN_GOALS = [
    # (i) Far fingertip: reach fingertip position (x, y) = (√2, √2),
    # twice as far as any training goal.
    {
        "label": "Far fingertip",
        "goal": [0.0, 0.0, 0.0, 0.0, float(jnp.sqrt(2)), float(jnp.sqrt(2))],
        "mask": [0, 0, 0, 0, 1, 1],
        "reward_type": "sparse", "a": 0.2, "sigma": 0.1,
        "terminate_on_goal": True,
    },
    # (ii) Target angle: reach joint configuration θ = (0, π). OOD goal
    # *type* — training goals only rewarded fingertip proximity, so this
    # tests whether the WM's dynamics knowledge transfers to a new task
    # definition.
    {
        "label": "Target angle",
        "goal": [float(0.0), float(jnp.pi), 0.0, 0.0, 0.0, 0.0],
        "mask": [1, 1, 0, 0, 0, 0],
        "reward_type": "sparse", "a": 0.2, "sigma": 0.1,
        "terminate_on_goal": True,
    },
    # (iii) Target velocity: reach ω = (2, 2). Pure-spin task — no
    # position constraint, just sustained angular velocity.
    {
        "label": "Target velocity",
        "goal": [0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
        "mask": [0, 0, 1, 1, 0, 0],
        "reward_type": "sparse", "a": 0.2, "sigma": 0.1,
        "terminate_on_goal": True,
    },
]

UNSEEN_NUM_EVAL_STARTS = 512

# ── Env / eval metadata ───────────────────────────────────────────────────────

EVAL_DIMS_2D = [(4, 5)]  # fingertip (fp_x, fp_y) plane for trajectory plots

# Slice point for WM dynamics plots: both joints at 45°, zero velocities.
_COS45 = float(jnp.cos(jnp.pi / 4))
_SIN45 = float(jnp.sin(jnp.pi / 4))
EVAL_SLICE_POINT = {
    "t1": float(jnp.pi / 4), "t2": float(jnp.pi / 4),
    "w1": 0.0, "w2": 0.0,
    "fp_x": 2 * _COS45, "fp_y": 2 * _SIN45,
}

# Which goals to overlay on the trajectory_policy plot. Defaults to all 4
# cardinal goals (the current goal grid has exactly NUM_GOALS=4).
WM_TRAJECTORY_GOAL_INDICES = [0, 1, 2, 3]



ENV_CONFIG = {
    "ENV_NAME": ENV_NAME,
    "STATE_DIM": STATE_DIM,
    "STATE_LABELS": STATE_LABELS,
    "STATE_RANGES": STATE_RANGES,
    "ACTION_DIM": ACTION_DIM,
    "ACTION_NAMES": ACTION_NAMES,
    "EVAL_DIMS_2D": EVAL_DIMS_2D,
    "VI_GRID_RES": VI_GRID_RES,
    "Q_EVAL_ON_TRAIN_SUPPORT": Q_EVAL_ON_TRAIN_SUPPORT,
    "VI_MAX_ITER": VI_MAX_ITER,
    "VI_STATE_DIM": VI_STATE_DIM,
    "VI_STATE_RANGES": VI_STATE_RANGES,
    "VI_STATE_LABELS": VI_STATE_LABELS,
    "WM_TRAJECTORY_GOAL_INDICES": WM_TRAJECTORY_GOAL_INDICES,
    "EVAL_SLICE_POINT": EVAL_SLICE_POINT,
    "UNSEEN_GOALS": UNSEEN_GOALS,
    "UNSEEN_NUM_EVAL_STARTS": UNSEEN_NUM_EVAL_STARTS,
}
