from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from utils.interactions import vmap_rollout_traj_node
from tqdm import tqdm

import matplotlib.pyplot as plt
from utils.visualization import plot_2_i_dq_comparison
from utils.evaluation import rollout_traj_node


@eqx.filter_value_and_grad
def grad_loss(model, true_obs, actions, tau, featurize):

    feat_pred_obs = vmap_rollout_traj_node(model, featurize, true_obs[:, 0, :], actions, tau)
    # create vmap_rollout_traj_node

    feat_true_obs = jax.vmap(jax.vmap(featurize, in_axes=(0)), in_axes=(0))(true_obs)
    # eventually vmap along multiple dimensions (multiple vmaps)

    return jnp.mean((feat_pred_obs - feat_true_obs) ** 2)


@eqx.filter_jit
def make_step(model, observations, actions, tau, featurize, opt_state, optim, loss_func):
    loss, grads = loss_func(model, observations, actions, tau, featurize)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def data_generation(data_gen_single, sequence_len, rng):
    observations, actions, key = jax.vmap(data_gen_single, in_axes=(0, None))(rng, sequence_len)
    return observations, actions, key


def fit(
    model,
    tau,
    featurize,
    train_steps,
    sequence_len,
    train_data_gen_sin,
    validation,
    rng,
    optim,
    init_opt_state,
    validate_every,
    loss_func,
):
    key = rng
    model_state = model
    opt_state = init_opt_state
    train_losses = []
    val_losses = []

    for i in tqdm(range(train_steps)):

        observations, actions, key = data_generation(train_data_gen_sin, sequence_len, key)

        model_state, opt_state, loss = make_step(
            model_state, observations, actions, tau, featurize, opt_state, optim, loss_func
        )

        train_losses.append(loss)

        if validate_every is not None and i % validate_every == 0:
            val_loss, terminate_learning = validation(model_state, key)
            val_losses.append(val_loss)
            if terminate_learning:
                break

    return model_state, opt_state, key, train_losses, val_losses


class ModelTrainer(eqx.Module):
    train_steps: jnp.int32
    batch_size: jnp.int32
    sequence_len: jnp.int32
    featurize: Callable
    train_data_gen_sin: Callable
    validation: Callable
    model_optimizer: optax._src.base.GradientTransformationExtraArgs
    tau: jnp.float32
    loss_func: Callable = grad_loss

    def fit(self, model, opt_state, key, validate_every=None):
        assert self.batch_size == key.shape[0]
        final_model, final_opt_state, final_key, train_losses, val_losses = fit(
            model,
            self.tau,
            self.featurize,
            self.train_steps,
            self.sequence_len,
            self.train_data_gen_sin,
            self.validation,
            key,
            self.model_optimizer,
            opt_state,
            validate_every,
            self.loss_func,
        )
        return final_model, final_opt_state, final_key, train_losses, val_losses
