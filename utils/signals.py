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


def steps_ref_traj(env, reset_fun, key, ref_len, step_lens=[100, 400]):
    t = 0
    ref = []

    while t < ref_len:
        key, subkey1, subkey2 = jax.random.split(key, num=3)
        ref_obs = reset_fun(env, subkey1)

        t_step = jax.random.randint(subkey2, shape=(1,), minval=step_lens[0], maxval=step_lens[1])

        ref.append(jnp.repeat(ref_obs[:, None], t_step, axis=1))
        t += t_step.item()

    return jnp.hstack(ref)[:, :ref_len].T
