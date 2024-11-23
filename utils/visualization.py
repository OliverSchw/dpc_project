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


def plot_2_i_dq_ref_tracking_time(obs1, obs2, obs_ref, axes, tau=1e-4):
    if obs1.shape[0] > obs_ref.shape[0]:
        # cut off initial state
        obs1 = obs1[1:]
    if obs2.shape[0] > obs_ref.shape[0]:
        # cut off initial state
        obs2 = obs2[1:]
    assert obs1.shape[0] == obs2.shape[0] == obs_ref.shape[0]
    time = jnp.linspace(0, obs_ref.shape[0] - 1, obs_ref.shape[0]) * tau
    axes[0].plot(time, obs_ref[:, 0], label="i_d_ref")  # ,label="currents"
    axes[0].plot(time, obs1[:, 0], label="i_d_1")  # ,label="currents"
    axes[0].plot(time, obs2[:, 0], label="i_d_2")  # ,label="currents"
    axes[1].plot(time, obs_ref[:, 1], label="i_q_ref")  # ,label="currents"
    axes[1].plot(time, obs1[:, 1], label="i_q_1")
    axes[1].plot(time, obs2[:, 1], label="i_q_2")  #
    axes[1].set_ylim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_ylabel("i_d")
    axes[1].set_ylabel("i_q")
    axes[1].set_xlabel("time")
    axes[0].legend()
    axes[1].legend()
    return


def plot_2_i_dq_comparison(obs1, obs2, axes, tau=1e-4):
    assert obs1.shape[0] == obs2.shape[0]
    time = jnp.linspace(0, obs1.shape[0] - 1, obs1.shape[0]) * tau
    axes[0].plot(time, obs1[:, 0], label="i_d_1")  # ,label="currents"
    axes[0].plot(time, obs2[:, 0], label="i_d_2")  # ,label="currents"
    axes[1].plot(time, obs1[:, 1], label="i_q_1")
    axes[1].plot(time, obs2[:, 1], label="i_q_2")  #
    axes[1].set_ylim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_ylabel("i_d")
    axes[1].set_ylabel("i_q")
    axes[1].set_xlabel("time in s")
    axes[0].legend()
    axes[1].legend()
    return
