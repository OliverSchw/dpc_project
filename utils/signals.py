from typing import Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax
import optax

from tqdm import tqdm


@eqx.filter_jit
def generate_n_random_numbers_with_sum_M(key, n, M):
    key, subkey = jax.random.split(key)

    # Generate n-1 random "cut points" between 0 and M, sort them
    cuts = jnp.sort(jax.random.randint(subkey, shape=((n - 1),), minval=1, maxval=M - 1))

    # Create the list of numbers by taking differences between the cut points
    numbers = jnp.concatenate([cuts[0][None], (cuts[1 : n - 1] - cuts[0 : n - 2]), (M - cuts[-1][None])])
    return numbers, key
