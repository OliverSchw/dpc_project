from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


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


class MLP_tanh(eqx.Module):
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
        return jax.nn.hard_tanh(self.layers[-1](x))


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
