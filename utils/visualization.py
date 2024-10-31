from typing import Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax
import optax

from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_i_dq_ref_tracking_time(obs, obs_ref, axes, tau=1e-4):
    if obs.shape[0] > obs_ref.shape[0]:
        # cut off initial state
        obs = obs[1:]
    assert obs.shape[0] == obs_ref.shape[0]
    time = jnp.linspace(0, obs_ref.shape[0] - 1, obs_ref.shape[0]) * tau
    axes[0].plot(time, obs_ref[:, 0], label="i_d_ref")  # ,label="currents"
    axes[0].plot(time, obs[:, 0], label="i_d")  # ,label="currents"
    axes[1].plot(time, obs_ref[:, 1], label="i_q_ref")  # ,label="currents"
    axes[1].plot(time, obs[:, 1], label="i_q")  #
    axes[1].set_ylim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_ylabel("i_d")
    axes[1].set_ylabel("i_q")
    axes[1].set_xlabel("time")
    axes[0].legend()
    axes[1].legend()
    return
