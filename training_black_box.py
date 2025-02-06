import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

import equinox as eqx
import diffrax
import optax

from policy.policy_training import DPCTrainer
from exciting_environments.pmsm.pmsm_env import PMSM
from policy.networks import MLP  # ,MLP2
import matplotlib.pyplot as plt

from utils.interactions import rollout_traj_env_policy
from models.model_training import ModelTrainer
from models.models import NeuralEulerODE

gpus = jax.devices()
jax.config.update("jax_default_device", gpus[0])


import json

with open("model_data/dmpe4.json") as json_data:
    d = json.load(json_data)
long_obs = jnp.array(d["observations"])
long_acts = jnp.array(d["actions"])


def step_eps(eps, omega_el, tau, tau_scale=1.0):
    eps += omega_el * tau * tau_scale
    eps %= 2 * jnp.pi
    boolean = eps > jnp.pi
    summation_mask = boolean * -2 * jnp.pi
    eps = eps + summation_mask
    return eps


eps = [0]
for i in range(long_obs.shape[0]):
    eps.append(step_eps(eps[-1], 3 * 1500 / 60 * 2 * jnp.pi, 1e-4))
cos_long_eps = jnp.cos(jnp.array(eps[:-1])[:, None])
sin_long_eps = jnp.sin(jnp.array(eps[:-1])[:, None])
long_obs = jnp.hstack([long_obs, cos_long_eps, sin_long_eps])

long_obs_train = long_obs[:-400]
long_acts_train = long_acts[:-400]
long_obs_val = long_obs[-400:]
long_acts_val = long_acts[-399:]


def data_gen_single(rng, sequence_len):
    rng, subkey = jax.random.split(rng)
    idx = jax.random.randint(subkey, shape=(1,), minval=0, maxval=(long_obs_train.shape[0] - sequence_len - 1))

    slice = jnp.linspace(start=idx, stop=idx + sequence_len, num=sequence_len + 1, dtype=int).T
    act_slice = jnp.linspace(start=idx, stop=idx + sequence_len - 1, num=sequence_len, dtype=int).T

    obs = long_obs_train[slice][0]
    acts = long_acts_train[act_slice][0]
    return obs, acts, rng


from utils.interactions import rollout_traj_node


# @eqx.filter_jit
def validation(model_state, rng):
    feat_pred_obs = jax.vmap(rollout_traj_node, in_axes=(None, None, 0, 0, None))(
        model_state, featurize_node, long_obs_val[:-1, :], long_acts_val[:, None], 1e-4
    )
    feat_pred_obs = feat_pred_obs[:, 1, :]
    feat_true_obs = jax.vmap(featurize_node, in_axes=(0))(long_obs_val[1:])
    terminate = False
    val_loss = jnp.mean(
        (feat_pred_obs[:, :2] - feat_true_obs[:, :2]) ** 2
    )  # validate only on i_dq and not cos(eps) and sin(eps)
    # print(val_loss)
    return val_loss, terminate


# @eqx.filter_jit
def validation_eps(model_state, rng):
    feat_pred_obs = jax.vmap(rollout_traj_node, in_axes=(None, None, 0, 0, None))(
        model_state, featurize_node_eps, long_obs_val[:-1, :], long_acts_val[:, None], 1e-4
    )
    feat_pred_obs = feat_pred_obs[:, 1, :]
    feat_true_obs = jax.vmap(featurize_node_eps, in_axes=(0))(long_obs_val[1:])
    terminate = False
    val_loss = jnp.mean(
        (feat_pred_obs[:, :2] - feat_true_obs[:, :2]) ** 2
    )  # validate only on i_dq and not cos(eps) and sin(eps)
    # print(val_loss)
    return val_loss, terminate


def train_model(model, featurize, validation):
    optimizer_node = optax.adam(1e-4)
    batch_size = 100
    mtrainer = ModelTrainer(
        train_steps=1_000_000,
        batch_size=batch_size,
        sequence_len=1,
        featurize=featurize,
        train_data_gen_sin=data_gen_single,
        validation=validation,
        model_optimizer=optimizer_node,
        tau=1e-4,
    )
    keys = jax.vmap(jax.random.PRNGKey)(np.random.randint(0, 2**31, size=(batch_size,)))
    opt_state = optimizer_node.init(model)
    fin_node, fin_opt_state, fin_keys, losses, val_losses = mtrainer.fit(model, opt_state, keys, validate_every=10_000)
    return fin_node, losses, val_losses


def featurize_node_eps(obs):
    return obs[:4]


def featurize_node(obs):
    return obs[:2]


if __name__ == "__main__":
    key_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(len(key_idx)):
        key = jax.random.PRNGKey(key_idx[i])
        node = NeuralEulerODE([6, 128, 128, 128, 128, 4], key=key)
        fin_node_eps, losses_eps, val_losses_eps = train_model(node, featurize_node_eps, validation_eps)
        eqx.tree_serialise_leaves(
            f"final_models/black_box/Model_eps_4_128_{key_idx[i]}.eqx",
            fin_node_eps,
        )
        jnp.save(
            f"final_models/black_box/losses/Model_eps_4_128_{key_idx[i]}.npy",
            losses_eps,
        )
        jnp.save(
            f"final_models/black_box/val_losses/Model_eps_4_128_{key_idx[i]}.npy",
            val_losses_eps,
        )
        node = NeuralEulerODE([4, 128, 128, 128, 128, 2], key=key)
        fin_node, losses, val_losses = train_model(node, featurize_node, validation)
        eqx.tree_serialise_leaves(
            f"final_models/black_box/Model_4_128_{key_idx[i]}.eqx",
            fin_node,
        )
        jnp.save(
            f"final_models/black_box/losses/Model_4_128_{key_idx[i]}.npy",
            losses,
        )
        jnp.save(
            f"final_models/black_box/val_losses/Model_4_128_{key_idx[i]}.npy",
            val_losses,
        )
        node = NeuralEulerODE([6, 64, 64, 64, 4], key=key)
        fin_node_eps, losses_eps, val_losses_eps = train_model(node, featurize_node_eps, validation_eps)
        eqx.tree_serialise_leaves(
            f"final_models/black_box/Model_eps_3_64_{key_idx[i]}.eqx",
            fin_node_eps,
        )
        jnp.save(
            f"final_models/black_box/losses/Model_eps_3_64_{key_idx[i]}.npy",
            losses_eps,
        )
        jnp.save(
            f"final_models/black_box/val_losses/Model_eps_3_64_{key_idx[i]}.npy",
            val_losses_eps,
        )
        node = NeuralEulerODE([4, 64, 64, 64, 2], key=key)
        fin_node, losses, val_losses = train_model(node, featurize_node, validation)
        eqx.tree_serialise_leaves(
            f"final_models/black_box/Model_3_64_{key_idx[i]}.eqx",
            fin_node,
        )
        jnp.save(
            f"final_models/black_box/losses/Model_3_64_{key_idx[i]}.npy",
            losses,
        )
        jnp.save(
            f"final_models/black_box/val_losses/Model_3_64_{key_idx[i]}.npy",
            val_losses,
        )
        node = NeuralEulerODE([6, 32, 32, 4], key=key)
        fin_node_eps, losses_eps, val_losses_eps = train_model(node, featurize_node_eps, validation_eps)
        eqx.tree_serialise_leaves(
            f"final_models/black_box/Model_eps_2_32_{key_idx[i]}.eqx",
            fin_node_eps,
        )
        jnp.save(
            f"final_models/black_box/losses/Model_eps_2_32_{key_idx[i]}.npy",
            losses_eps,
        )
        jnp.save(
            f"final_models/black_box/val_losses/Model_eps_2_32_{key_idx[i]}.npy",
            val_losses_eps,
        )
        node = NeuralEulerODE([4, 32, 32, 2], key=key)
        fin_node, losses, val_losses = train_model(node, featurize_node, validation)
        eqx.tree_serialise_leaves(
            f"final_models/black_box/Model_2_32_{key_idx[i]}.eqx",
            fin_node,
        )
        jnp.save(
            f"final_models/black_box/losses/Model_2_32_{key_idx[i]}.npy",
            losses,
        )
        jnp.save(
            f"final_models/black_box/val_losses/Model_2_32_{key_idx[i]}.npy",
            val_losses,
        )
