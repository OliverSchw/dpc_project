from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import json


class MLP_lin(eqx.Module):
    # TODO different output layers
    layers: list[eqx.nn.Linear]

    def __init__(self, layer_sizes, key):
        self.layers = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.leaky_relu(layer(x))
        return self.layers[-1](x)


class MLP(eqx.Module):

    layers: list[eqx.nn.Linear]
    output_activation: Callable = eqx.field(static=True)
    hidden_activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        layer_sizes,
        key,
        hidden_activation: Callable = jax.nn.leaky_relu,
        output_activation: Callable = lambda x: x,
    ):

        self.layers = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey))
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def __call__(self, x):

        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        return self.output_activation(self.layers[-1](x))


class NeuralEulerODE(eqx.Module):
    func: MLP_lin

    def __init__(self, layer_sizes, key, **kwargs):
        super().__init__(**kwargs)
        self.func = MLP_lin(layer_sizes=layer_sizes, key=key)

    def step(self, obs, action, tau):
        obs_act = jnp.hstack([obs, action])
        next_obs = obs + tau * self.func(obs_act)
        return next_obs

    def __call__(self, init_obs, actions, tau):

        def body_fun(carry, action):
            obs = carry
            obs = self.step(obs, action, tau)
            return obs, obs

        _, observations = jax.lax.scan(body_fun, init_obs, actions)
        observations = jnp.concatenate([init_obs[None, :], observations], axis=0)
        return observations


def save_model(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(filename, model_class):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        hyperparams["key"] = jnp.array(hyperparams["key"], dtype=jnp.uint32)
        model = model_class(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)
