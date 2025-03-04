import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

import equinox as eqx
import diffrax
import optax

import os
from policy.policy_training import DPCTrainer
from exciting_environments.pmsm.pmsm_env import PMSM
from policy.networks import MLP  # ,MLP2
from utils.interactions import rollout_traj_env_policy
from models.model_training import ModelTrainer
from models.models import NeuralEulerODE

gpus = jax.devices()
jax.config.update("jax_default_device", gpus[0])


@eqx.filter_jit
def reset_env(env, rng):
    obs, _ = env.reset(env.env_properties, rng)  #
    obs = obs.at[2].set((3 * 1500 / 60 * 2 * jnp.pi) / (2 * jnp.pi * 3 * 11000 / 60))
    return obs


@eqx.filter_jit
def motor_env_dat_gen_sin(env, reset_env, rng, traj_len):
    rng, subkey = jax.random.split(rng)
    ref_obs = reset_env(env, subkey)
    rng, subkey = jax.random.split(rng)
    init_obs = reset_env(env, subkey)
    return init_obs, ref_obs, rng


@eqx.filter_jit
def featurize(obs, ref_obs, featurize_state=jnp.array([0, 0])):
    feat_obs = jnp.concatenate(
        [obs[0:2], ref_obs[0:2], ref_obs[0:2] - obs[0:2], featurize_state]
    )  # jnp.concatenate([obs[0:2],obs[6:8],ref_obs[0:2],ref_obs[0:2]-obs[0:2],featurize_state])
    featurize_state = jnp.clip(featurize_state + ref_obs[0:2] - obs[0:2], min=-1, max=1) * (
        jnp.sign(0.01 - jnp.sum((ref_obs[0:2] - obs[0:2]) ** 2)) * 0.5 + 0.5
    )
    return feat_obs, featurize_state


@eqx.filter_jit
def mse_loss(feat_obs):
    loss = jnp.mean(jnp.sum((feat_obs[:, 4:6]) ** 2, axis=1))  # be aware of idx if changing featurize
    return loss


@eqx.filter_jit
def penalty_loss(feat_obs):
    loss = jnp.array([0])
    return loss


def train_policy():
    motor_env = PMSM(
        saturated=True,
        LUT_motor_name="BRUSA",
        batch_size=1,
        control_state=[],
        static_params={
            "p": 3,
            "r_s": 15e-3,
            "l_d": 0.37e-3,
            "l_q": 1.2e-3,
            "psi_p": 65.6e-3,
            "deadtime": 0,
        },
    )
    jax_key = jax.random.PRNGKey(2)
    policy = MLP([8, 64, 64, 64, 2], key=jax_key)
    optimizer = optax.adam(5e-4)
    opt_state = optimizer.init(policy)
    data_batch_size = 100
    trainer = DPCTrainer(
        batch_size=data_batch_size,
        train_steps=1000,
        horizon_length=50,
        reset_env=reset_env,
        data_gen_sin=motor_env_dat_gen_sin,
        featurize=featurize,
        policy_optimizer=optimizer,
        ref_loss=mse_loss,
        constr_penalty=penalty_loss,
        ref_loss_weight=1,
    )
    keys = jax.vmap(jax.random.PRNGKey)(np.random.randint(0, 2**31, size=(data_batch_size,)))
    fin_policy, fin_opt_state, fin_keys, losses = trainer.fit_non_jit(policy, motor_env, keys, opt_state)
    return fin_policy


def featurize_node(obs):
    return obs[:2]


def train_node(jax_key, policy, batch_size, train_steps, sequence_length):
    motor_env = PMSM(
        saturated=True,
        LUT_motor_name="BRUSA",
        batch_size=1,
        control_state=[],
        static_params={
            "p": 3,
            "r_s": 15e-3,
            "l_d": 0.37e-3,
            "l_q": 1.2e-3,
            "psi_p": 65.6e-3,
            "deadtime": 0,
        },
    )

    def data_gen_single(rng, sequence_len):
        rng, subkey = jax.random.split(rng)
        ref_obs = reset_env(motor_env, subkey)
        rng, subkey = jax.random.split(rng)
        init_obs = reset_env(motor_env, subkey)
        obs, acts = rollout_traj_env_policy(policy, init_obs, ref_obs, sequence_len, motor_env, featurize)
        return obs, acts, rng

    node = NeuralEulerODE([4, 128, 128, 128, 2], key=jax_key)
    optimizer_node = optax.adam(5e-4)
    opt_state = optimizer_node.init(node)
    mtrainer = ModelTrainer(
        train_steps=train_steps,
        batch_size=batch_size,
        sequence_len=sequence_length,
        featurize=featurize_node,
        data_gen_sin=data_gen_single,
        model_optimizer=optimizer_node,
        tau=motor_env.tau,
    )
    keys = jax.vmap(jax.random.PRNGKey)(np.random.randint(0, 2**31, size=(batch_size,)))
    fin_node, fin_opt_state, fin_keys, losses = mtrainer.fit_non_jit(node, opt_state, keys)
    return fin_node, losses


if __name__ == "__main__":
    policy = train_policy()
    seq_lens = [20, 50, 100]
    train_stepss = [500000, 250000, 100000]
    for i in range(3):
        key = jax.random.PRNGKey(i)
        batch_size = 1000
        train_steps = train_stepss[i]
        seq_len = seq_lens[i]
        fin_node, losses = train_node(key, policy, batch_size, train_steps, seq_len)
        eqx.tree_serialise_leaves(
            f"trained_models/long_training/Model_{int(train_steps/1000)}k_{seq_len}len_{batch_size}b_step0_0005_number{i}.eqx",
            fin_node,
        )
        jnp.save(
            f"trained_models/long_training/losses/Model_{int(train_steps/1000)}k_{seq_len}len_{batch_size}b_step0_0005_number{i}.npy",
            losses,
        )
