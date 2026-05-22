"""JSON IO for the config dicts that get materialised at run time.

Pairs with the run.import_config_module Python-module loader: that one
imports a .py config file and reads its uppercase globals into a dict;
this module persists / restores such dicts (PQN_CONFIG, WM_CONFIG,
ENV_CONFIG) as JSON next to each phase's checkpoint so callers can rebuild
the network / WM later without re-loading the original .py.
"""

import json
import os

import jax.numpy as jnp
import numpy as np


def save_config(config, path):
    """Save config dict as human-readable JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialized = {}
    for k, v in config.items():
        if isinstance(v, (jnp.ndarray, np.ndarray)):
            serialized[k] = np.array(v).tolist()
        elif isinstance(v, tuple):
            serialized[k] = list(v)
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serialized[k] = v
        else:
            serialized[k] = "<not serializable>"
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)


def load_config(path):
    """Load config dict from JSON, converting nested lists back to jnp arrays.

    Only nested lists (tensors like GOALS, STATE_RANGES) and lists of floats
    are coerced. Plain int lists (e.g. ACTOR_HIDDEN_SIZES=[512,256,128],
    GOAL_INPUT_DIMS=[6,7]) stay as Python lists so Flax Modules and indexers
    that expect concrete Python ints keep working.
    """
    with open(path) as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, list) or isinstance(first, float):
                cfg[k] = jnp.array(v)
    return cfg


def serialize_for_pickle(config):
    """Convert config values to pickle-safe types (JAX arrays → lists).

    Used by PQN checkpoint saver to ensure embedded config survives pickle.
    """
    out = {}
    for k, v in config.items():
        if isinstance(v, jnp.ndarray):
            out[k] = np.array(v).tolist()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
