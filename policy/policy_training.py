from typing import Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax
import optax

from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.visualization import plot_2_i_dq_ref_tracking_time, plot_i_dq_ref_tracking_time
from utils.interactions import (
    vmap_rollout_traj_env_policy,
    vmap_rollout_traj_node_policy,
    rollout_traj_env_policy,
    rollout_traj_node_policy,
    rollout_traj_node,
    vmap_rollout_traj_env,
)
from models.model_training import make_step as make_step_train_node

# loss construct from (loss and soft_pen) (done)
# rollout (done)
# data_generation (done)
# fit (done)


@eqx.filter_value_and_grad
def grad_loss(policy, env, init_obs, ref_obs, horizon_length, featurize, ref_loss_fun, penalty_fun, ref_loss_weight=1):
    obs, acts = vmap_rollout_traj_env_policy(policy, init_obs, ref_obs, horizon_length, env, featurize)
    loss = vmap_compute_loss(obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting=ref_loss_weight)
    return loss


@eqx.filter_value_and_grad
def grad_loss_node(
    policy,
    node,
    tau,
    featurize_node,
    init_obs,
    ref_obs,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    ref_loss_weight=1,
):
    obs, acts = vmap_rollout_traj_node_policy(
        policy, node, tau, init_obs, ref_obs, horizon_length, featurize, featurize_node
    )
    loss = vmap_compute_loss(obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting=ref_loss_weight)
    return loss


@eqx.filter_jit
def make_step(
    policy,
    env,
    init_obs,
    ref_obs,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    optim,
    opt_state,
    ref_loss_weight=1,
):
    loss, grads = grad_loss(
        policy,
        env,
        init_obs,
        ref_obs,
        horizon_length,
        featurize,
        ref_loss_fun,
        penalty_fun,
        ref_loss_weight=ref_loss_weight,
    )
    updates, opt_state = optim.update(grads, opt_state)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss


@eqx.filter_jit
def make_step_node(
    policy,
    node,
    tau,
    featurize_node,
    init_obs,
    ref_obs,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    optim,
    opt_state,
    ref_loss_weight=1,
):
    loss, grads = grad_loss_node(
        policy,
        node,
        tau,
        featurize_node,
        init_obs,
        ref_obs,
        horizon_length,
        featurize,
        ref_loss_fun,
        penalty_fun,
        ref_loss_weight=ref_loss_weight,
    )
    updates, opt_state = optim.update(grads, opt_state)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss


@eqx.filter_value_and_grad
def grad_loss_node_weight(
    policy,
    node,
    tau,
    featurize_node,
    init_obs,
    ref_obs,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    node_loss,
    ref_loss_weight=1,
):
    obs, acts = vmap_rollout_traj_node_policy(
        policy, node, tau, init_obs, ref_obs, horizon_length, featurize, featurize_node
    )
    loss = vmap_compute_loss(obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting=ref_loss_weight)
    weighted_loss = loss * jnp.exp(-1000 * node_loss)
    return weighted_loss


@eqx.filter_jit
def make_step_node_weight(
    policy,
    node,
    tau,
    featurize_node,
    init_obs,
    ref_obs,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    optim,
    opt_state,
    node_loss,
    ref_loss_weight=1,
):
    loss, grads = grad_loss_node_weight(
        policy,
        node,
        tau,
        featurize_node,
        init_obs,
        ref_obs,
        horizon_length,
        featurize,
        ref_loss_fun,
        penalty_fun,
        node_loss,
        ref_loss_weight=ref_loss_weight,
    )
    updates, opt_state = optim.update(grads, opt_state)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss


@eqx.filter_jit
def compute_loss(sim_obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting=0.9):
    feat_obs, _ = jax.vmap(featurize, in_axes=(0, None))(sim_obs, ref_obs)
    # change if ref_obs not constant (scalar) anymore
    ref_loss = ref_loss_fun(feat_obs)
    penalty_loss = penalty_fun(feat_obs)
    loss = (weighting) * ref_loss + (1 - weighting) * penalty_loss
    loss = jnp.clip(loss, max=1e5)
    return loss


@eqx.filter_jit
def vmap_compute_loss(sim_obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting=0.9):
    loss = jax.vmap(compute_loss, in_axes=(0, 0, None, None, None, None))(
        sim_obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting
    )
    loss = jnp.mean(loss)
    return loss


@eqx.filter_jit
def data_generation(env, reset_env, data_gen_single, rng, traj_len=None):
    # TODO implement ref_traj other than constants -> traj_len
    init_obs, ref_obs, key = jax.vmap(data_gen_single, in_axes=(None, None, 0, None))(env, reset_env, rng, traj_len)
    return init_obs, ref_obs, key


def fit_on_env_non_jit(
    policy,
    train_steps,
    env,
    reset_env,
    data_gen_sin,
    rng,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    optim,
    init_opt_state,
    ref_loss_weight=1,
):
    key = rng
    policy_state = policy
    opt_state = init_opt_state
    losses = []

    for i in tqdm(range(train_steps)):

        init_obs, ref_obs, key = data_generation(env, reset_env, data_gen_sin, key)

        policy_state, opt_state, loss = make_step(
            policy_state,
            env,
            init_obs,
            ref_obs,
            horizon_length,
            featurize,
            ref_loss_fun,
            penalty_fun,
            optim,
            opt_state,
            ref_loss_weight=ref_loss_weight,
        )

        losses.append(loss)
    return policy_state, opt_state, key, losses


def fit_on_node_non_jit(
    policy,
    train_steps,
    node,
    tau,
    featurize_node,
    reset_env,
    data_gen_sin,
    rng,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    optim,
    init_opt_state,
    ref_loss_weight=1,
):
    key = rng
    policy_state = policy
    opt_state = init_opt_state
    losses = []

    for i in tqdm(range(train_steps)):

        init_obs, ref_obs, key = data_generation(node, reset_env, data_gen_sin, key)

        policy_state, opt_state, loss = make_step_node(
            policy_state,
            node,
            tau,
            featurize_node,
            init_obs,
            ref_obs,
            horizon_length,
            featurize,
            ref_loss_fun,
            penalty_fun,
            optim,
            opt_state,
            ref_loss_weight,
        )
        losses.append(loss)
    return policy_state, opt_state, key, losses


def data_slice(rng, obs_long, acts_long, sequence_len):
    rng, subkey = jax.random.split(rng)
    idx = jax.random.randint(subkey, shape=(1,), minval=0, maxval=(obs_long.shape[0] - sequence_len - 1))

    slice = jnp.linspace(start=idx, stop=idx + sequence_len, num=sequence_len + 1, dtype=int).T
    act_slice = jnp.linspace(start=idx, stop=idx + sequence_len - 1, num=sequence_len, dtype=int).T

    obs = obs_long[slice][0]
    acts = acts_long[act_slice][0]
    return obs, acts, rng


def fit_policy_and_env(
    policy,
    train_steps,
    env,
    node,
    tau,
    featurize_node,
    reset_env,
    data_gen_sin,
    rng,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    pol_optim,
    node_optim,
    pol_init_opt_state,
    node_init_opt_state,
    val_data_gen_sin,
    plot_every,
    ref_loss_weight=1,
):
    key = rng
    node_state = node
    policy_state = policy
    node_opt_state = node_init_opt_state
    policy_opt_state = pol_init_opt_state
    policy_losses = []
    node_losses = []

    for i in tqdm(range(train_steps)):

        init_obs, ref_obs, key = data_generation(env, reset_env, data_gen_sin, key)

        # discuss wether case 1: policy acts directly on env
        #               case 2: policy acts on node and resulting actions applied to env
        # observations, actions = vmap_rollout_traj_node_policy(
        #     policy_state, init_obs, ref_obs, horizon_length, env, featurize
        # )

        obs, actions = vmap_rollout_traj_node_policy(
            policy_state, node_state, tau, init_obs, ref_obs, horizon_length, featurize, featurize_node
        )
        observations = vmap_rollout_traj_env(env, init_obs, actions)

        for j in range(25):
            obs_short, acts_short, key = jax.vmap(data_slice, in_axes=(0, 0, 0, None))(key, observations, actions, 1)

            node_state, node_opt_state, node_loss = make_step_train_node(
                node_state, obs_short, acts_short, tau, featurize_node, node_opt_state, node_optim
            )

        policy_state, policy_opt_state, policy_loss = make_step_node_weight(
            policy_state,
            node_state,
            tau,
            featurize_node,
            init_obs,
            ref_obs,
            horizon_length,
            featurize,
            ref_loss_fun,
            penalty_fun,
            pol_optim,
            policy_opt_state,
            node_loss,
            ref_loss_weight,
        )

        policy_losses.append(policy_loss)
        node_losses.append(node_loss)
        if i is not None and i % plot_every == 0:  # and i > 0
            val_init, val_ref = val_data_gen_sin(env, reset_env, rng, horizon_length)

            fig, axes = plt.subplots(2, 2, figsize=(12, 3), sharex=True)

            axes[0, 0].set_title("Policy on Node")
            axes[0, 1].set_title("Policy on Env")

            obs_node, acts_node = rollout_traj_node_policy(
                policy_state, node_state, tau, val_init, val_ref, val_ref.shape[0], featurize, featurize_node
            )
            plot_i_dq_ref_tracking_time(obs_node, val_ref, axes[:, 0])

            obs, acts = rollout_traj_env_policy(policy_state, val_init, val_ref, val_ref.shape[0], env, featurize)
            obs_node_roll = rollout_traj_node(node_state, featurize_node, val_init, acts, tau)
            plot_2_i_dq_ref_tracking_time(obs, obs_node_roll, val_ref, axes[:, 1], name1="env", name2="acts_on_node")

            plt.show()

    return policy_state, policy_opt_state, node_state, node_opt_state, key, policy_losses, node_losses


@eqx.filter_jit
def fit(
    policy,
    train_steps,
    env,
    reset_env,
    data_gen_sin,
    rng,
    horizon_length,
    featurize,
    ref_loss_fun,
    penalty_fun,
    optim,
    init_opt_state,
    ref_loss_weight=1,
):

    dynamic_init_policy_state, static_policy_state = eqx.partition(policy, eqx.is_inexact_array)
    init_carry = (dynamic_init_policy_state, init_opt_state, rng)

    def body_fun(i, carry):
        dynamic_policy_state, opt_state, key = carry
        policy_state = eqx.combine(static_policy_state, dynamic_policy_state)

        init_obs, ref_obs, key = data_generation(env, reset_env, data_gen_sin, key)

        new_policy_state, new_opt_state, _ = make_step(
            policy_state,
            env,
            init_obs,
            ref_obs,
            horizon_length,
            featurize,
            ref_loss_fun,
            penalty_fun,
            optim,
            opt_state,
            ref_loss_weight=ref_loss_weight,
        )
        new_dynamic_policy_state, new_static_policy_state = eqx.partition(new_policy_state, eqx.is_inexact_array)
        assert eqx.tree_equal(static_policy_state, new_static_policy_state) is True
        return (new_dynamic_policy_state, new_opt_state, key)

    final_dynamic_policy_state, final_opt_state, final_key = jax.lax.fori_loop(
        lower=0, upper=train_steps, body_fun=body_fun, init_val=init_carry
    )
    final_policy = eqx.combine(static_policy_state, final_dynamic_policy_state)
    return final_policy, final_opt_state, final_key


# DPCTrainer
class DPCTrainer(eqx.Module):
    batch_size: jnp.int32
    train_steps: jnp.int32
    horizon_length: jnp.int32
    reset_env: Callable
    data_gen_sin: Callable
    featurize: Callable
    policy_optimizer: optax._src.base.GradientTransformationExtraArgs
    ref_loss: Callable
    constr_penalty: Callable
    ref_loss_weight: jnp.float32

    # @eqx.filter_jit
    def fit_on_env(self, policy, env, key, opt_state):
        assert self.batch_size == key.shape[0]
        final_policy, final_opt_state, final_key = fit(
            policy=policy,
            train_steps=self.train_steps,
            env=env,
            reset_env=self.reset_env,
            data_gen_sin=self.data_gen_sin,
            rng=key,
            horizon_length=self.horizon_length,
            featurize=self.featurize,
            ref_loss_fun=self.ref_loss,
            penalty_fun=self.constr_penalty,
            optim=self.policy_optimizer,
            init_opt_state=opt_state,
            ref_loss_weight=self.ref_loss_weight,
        )

        return final_policy, final_opt_state, final_key, None

    def fit_non_jit(self, policy, env, key, opt_state):
        assert self.batch_size == key.shape[0]
        final_policy, final_opt_state, final_key, losses = fit_on_env_non_jit(
            policy=policy,
            train_steps=self.train_steps,
            env=env,
            reset_env=self.reset_env,
            data_gen_sin=self.data_gen_sin,
            rng=key,
            horizon_length=self.horizon_length,
            featurize=self.featurize,
            ref_loss_fun=self.ref_loss,
            penalty_fun=self.constr_penalty,
            optim=self.policy_optimizer,
            init_opt_state=opt_state,
            ref_loss_weight=self.ref_loss_weight,
        )

        return final_policy, final_opt_state, final_key, losses  #

    def fit_on_node_non_jit(self, policy, node, tau, featurize_node, key, opt_state):
        assert self.batch_size == key.shape[0]
        final_policy, final_opt_state, final_key, losses = fit_on_node_non_jit(
            policy=policy,
            train_steps=self.train_steps,
            node=node,
            tau=tau,
            featurize_node=featurize_node,
            reset_env=self.reset_env,
            data_gen_sin=self.data_gen_sin,
            rng=key,
            horizon_length=self.horizon_length,
            featurize=self.featurize,
            ref_loss_fun=self.ref_loss,
            penalty_fun=self.constr_penalty,
            optim=self.policy_optimizer,
            init_opt_state=opt_state,
            ref_loss_weight=self.ref_loss_weight,
        )
        return final_policy, final_opt_state, final_key, losses

    def fit_policy_and_env(
        self,
        policy,
        env,
        node,
        tau,
        featurize_node,
        key,
        pol_opt_state,
        node_optimizer,
        node_opt_state,
        val_data_gen_sin,
        plot_every,
    ):
        assert self.batch_size == key.shape[0]
        policy_state, policy_opt_state, node_state, node_opt_state, key, policy_losses, node_losses = (
            fit_policy_and_env(
                policy=policy,
                train_steps=self.train_steps,
                env=env,
                node=node,
                tau=tau,
                featurize_node=featurize_node,
                reset_env=self.reset_env,
                data_gen_sin=self.data_gen_sin,
                rng=key,
                horizon_length=self.horizon_length,
                featurize=self.featurize,
                ref_loss_fun=self.ref_loss,
                penalty_fun=self.constr_penalty,
                pol_optim=self.policy_optimizer,
                node_optim=node_optimizer,
                pol_init_opt_state=pol_opt_state,
                node_init_opt_state=node_opt_state,
                val_data_gen_sin=val_data_gen_sin,
                plot_every=plot_every,
                ref_loss_weight=self.ref_loss_weight,
            )
        )
        return policy_state, policy_opt_state, node_state, node_opt_state, key, policy_losses, node_losses
