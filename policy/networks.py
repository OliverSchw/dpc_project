import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax

from typing import Callable


class MLP(eqx.Module):

    layers: list[eqx.nn.Linear]

    def __init__(self, layer_sizes, key):
        self.layers = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.leaky_relu(layer(x))
        return jnp.tanh(self.layers[-1](x))


# class MLP2(eqx.Module):
#     mlp: eqx.nn.MLP

#     def __init__(self, in_dim, out_dim, width_size, depth, *, key, **kwargs):
#         super().__init__(**kwargs)
#         self.mlp = eqx.nn.MLP(
#             in_size=in_dim,
#             out_size=out_dim,
#             width_size=width_size,
#             depth=depth,
#             final_activation=jnn.hard_tanh,
#             key=key,
#         )
#         # activation=jnn.leaky_relu,

#     def __call__(self, feat_obs):
#         return self.mlp(feat_obs)
