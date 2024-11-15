from typing import Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax
import optax

from tqdm import tqdm
from utils.interactions import vmap_rollout_traj_env_policy, vmap_rollout_traj_node_policy

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


@eqx.filter_jit
def compute_loss(sim_obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting=0.9):
    feat_obs, _ = jax.vmap(featurize, in_axes=(0, None))(sim_obs, ref_obs)
    # change if ref_obs not constant (scalar) anymore
    ref_loss = ref_loss_fun(feat_obs)
    penalty_loss = penalty_fun(feat_obs)
    loss = (weighting) * ref_loss + (1 - weighting) * penalty_loss
    return loss


@eqx.filter_jit
def vmap_compute_loss(sim_obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting=0.9):
    loss = jax.vmap(compute_loss, in_axes=(0, 0, None, None, None, None))(
        sim_obs, ref_obs, featurize, ref_loss_fun, penalty_fun, weighting
    )
    loss = jnp.sum(loss)
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
