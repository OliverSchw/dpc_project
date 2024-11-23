from typing import Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax
import optax

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.visualization import plot_i_dq_ref_tracking_time, plot_2_i_dq_ref_tracking_time
from utils.signals import steps_ref_traj
from utils.interactions import rollout_traj_env, rollout_traj_node, rollout_traj_env_policy, rollout_traj_node_policy


def steps_eval(env, reset_fun, policy, featurize, key, ref_len, init_obs_key=None, plot=True, step_lens=[50, 200]):
    obs_ref = steps_ref_traj(env, reset_fun, key, ref_len, step_lens=step_lens)
    init_obs = reset_fun(env, init_obs_key)
    obs, acts = rollout_traj_env_policy(policy, init_obs, obs_ref, ref_len, env, featurize)
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
        plot_i_dq_ref_tracking_time(obs, obs_ref, axes, tau=env.tau)
    return obs, obs_ref, acts


def steps_eval_node(
    env,
    node,
    tau,
    reset_fun,
    policy,
    featurize_policy,
    featurize_node,
    key,
    ref_len,
    init_obs_key=None,
    step_lens=[50, 200],
):
    obs_ref = steps_ref_traj(env, reset_fun, key, ref_len, step_lens=step_lens)
    init_obs = reset_fun(env, init_obs_key)
    obs, acts = rollout_traj_node_policy(
        policy, node, tau, init_obs, obs_ref, ref_len, featurize_policy, featurize_node
    )
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    plot_i_dq_ref_tracking_time(obs, obs_ref, axes, tau=env.tau)
    return obs, obs_ref, acts


def rollout_comparison(env, node, tau, init_obs, obs_ref, actions, featurize_node):
    obs_env = rollout_traj_env(env, init_obs, actions)
    obs_node = rollout_traj_node(node, featurize_node, init_obs, actions, tau)
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    plot_2_i_dq_ref_tracking_time(obs_env, obs_node, obs_ref, axes)
    return obs_env, obs_node
