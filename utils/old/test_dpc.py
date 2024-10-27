import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from dataclasses import dataclass
import matplotlib.pyplot as plt


#from utils.dpc_controller import EnvParameters, TrainDPCParameters, train_dpc_controller , loss_fn
from utils.old.dpc_controller import EnvParameters, TrainDPCParameters
from utils.old.helpers import  test_random_initial_state

import exciting_environments as excenvs




@dataclass
class TestDPCParameters:
    #layer_sizes: list
    horizon_length: int


@dataclass
class TestResults:
    test_loss: list
    test_state_trajectories: list
    test_action_trajectories: list


        
        

def test_dpc_controller(policy , env_params, train_dpc_params, test_dpc_params, key_test):
    
    key = jax.random.PRNGKey(0)
    key_test = jax.random.PRNGKey(21)
    
    env = excenvs.make(env_params.name, batch_size=env_params.batch_size, l=env_params.l, m=env_params.m, tau=env_params.tau, max_torque=env_params.max_torque)
    
    
    ref_state = jnp.tile(jnp.array([[0, 0]]), (env_params.batch_size, 1)).astype(jnp.float32)
    
    horizon_length= test_dpc_params.horizon_length
    
    
    
    @eqx.filter_jit
    def loss_fn(policy, initial_state, ref_state, horizon_length):
        
        def generate_actions(carry, _):
            
            state, _ = carry
            
            #key, subkey = jax.random.split(key)
            
            #policy_params = jnp.concatenate([state, ref_state], axis=-1)
            #action = jax.vmap(policy)(policy_params)
            
            action = jax.vmap(policy)(state)
            
            next_state = jax.vmap(env._ode_exp_euler_step)(state, action, env.env_state_normalizer, env.action_normalizer, env.static_params)
            
            
            return (next_state, None), (next_state, action, state)
        
        
        (_, (predict_states, actions, initial_states)) = jax.lax.scan(generate_actions, (initial_state, None), None, horizon_length)
        
        mse = jnp.mean((predict_states - ref_state)**2)
        
        return mse, predict_states, actions, initial_states
    
    
    
    def evaluate_policy(policy, key, batch_size):
        
        key, subkey = jax.random.split(key)
        
        test_initial_states = test_random_initial_state(subkey, env_params.batch_size)
        
        loss, predicted_states, actions, initial_states = loss_fn(policy, test_initial_states, ref_state, horizon_length)
        
        return loss, predicted_states, actions, initial_states

    
    test_loss, test_predicted_states, test_actions, test_initial_states = evaluate_policy(policy, key_test, env_params.batch_size)

    test_results = TestResults(test_loss = test_loss, test_state_trajectories=test_predicted_states, test_action_trajectories=test_actions)
    
    
    return  test_results
    