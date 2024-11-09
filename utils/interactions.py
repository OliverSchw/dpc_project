from typing import Callable

import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx
import diffrax
import optax

from tqdm import tqdm


# @eqx.filter_jit
# def rollout_traj_env(policy, init_obs, ref_obs, horizon_length, env, featurize):
#     # TODO change for non constant refs
#     init_state = env.generate_state_from_observation(init_obs, env.env_properties)

#     def body_fun(carry, _):

#         obs, state = carry

#         policy_in = featurize(obs, ref_obs)

#         action = policy(policy_in)

#         obs, state = env.step(state, action, env.env_properties)

#         return (obs, state), (obs, action)

#     _, (observations, actions) = jax.lax.scan(body_fun, (init_obs, init_state), None, horizon_length)
#     observations = jnp.concatenate([init_obs[None, :], observations], axis=0)

#     return observations, actions


@eqx.filter_jit
def rollout_traj_env(policy, init_obs, ref_obs, horizon_length, env, featurize):
    init_state = env.generate_state_from_observation(init_obs, env.env_properties)

    if len(ref_obs.shape) == 1:
        ref_o = jnp.repeat(ref_obs[None, :], horizon_length, axis=0)
    else:
        ref_o = ref_obs
        assert ref_obs.shape[0] == horizon_length

    _, init_feat_state = featurize(init_obs, ref_o[0])
    init_feat_state = jnp.zeros_like(init_feat_state)

    def body_fun(carry, ref):

        obs, state, feat_state = carry

        policy_in, feat_state = featurize(obs, ref, feat_state)

        action = policy(policy_in)

        obs, state = env.step(state, action, env.env_properties)

        return (obs, state, feat_state), (obs, action)

    _, (observations, actions) = jax.lax.scan(body_fun, (init_obs, init_state, init_feat_state), ref_o, horizon_length)
    observations = jnp.concatenate([init_obs[None, :], observations], axis=0)

    return observations, actions


@eqx.filter_jit
def vmap_rollout_traj_env(policy, init_obs, ref_obs, horizon_length, env, featurize):
    observations, actions = jax.vmap(rollout_traj_env, in_axes=(None, 0, 0, None, None, None))(
        policy, init_obs, ref_obs, horizon_length, env, featurize
    )
    return observations, actions


def rollout_traj_node(init_obs, ref_obs, env, policy):
    raise (NotImplementedError)
