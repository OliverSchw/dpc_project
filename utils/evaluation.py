from typing import Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax
import optax

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.visualization import plot_i_dq_ref_tracking_time
from utils.signals import steps_ref_traj
from utils.interactions import rollout_traj_env


def steps_eval(env, reset_fun, policy, featurize, key, ref_len, init_obs_key=None, step_lens=[50, 200]):
    obs_ref = steps_ref_traj(env, reset_fun, key, ref_len, step_lens=step_lens)
    init_obs = reset_fun(env, init_obs_key)
    obs, acts = rollout_traj_env(policy, init_obs, obs_ref, ref_len, env, featurize)
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    plot_i_dq_ref_tracking_time(obs, obs_ref, axes, tau=env.tau)
    return obs, obs_ref, acts
