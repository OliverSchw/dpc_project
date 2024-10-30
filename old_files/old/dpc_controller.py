import os
import time
import jax
import optax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from dataclasses import dataclass

from utils.old.helpers import train_random_initial_state 

import exciting_environments as excenvs



@dataclass
class EnvParameters:
    name: str
    batch_size: int
    l: float
    m: float
    tau: float
    max_torque: float


@dataclass
class TrainDPCParameters:
    layer_sizes: list
    learning_rate: float
    optimizer_type: str
    epochs: int
    horizon_lengths: list

@dataclass
class TrainingResults:
    training_loss: list 
    state_trajectories: list
    action_trajectories: list
           



def train_dpc_controller(env_params, train_dpc_params, key):
    
    seed = 11
    jax_key = jax.random.PRNGKey(seed)
    
    class PolicyNetwork(eqx.Module):
        
        layers: list[eqx.nn.Linear]
        
        def __init__(self, layer_sizes, key):
            self.layers = []
            for (fan_in, fan_out) in zip(layer_sizes[:-1], layer_sizes[1:]):
                key, subkey = jax.random.split(key)
                self.layers.append(eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey))
        
        
        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = jax.nn.leaky_relu(layer(x))
            return jnp.tanh(self.layers[-1](x))

    
    
    @eqx.filter_jit
    #def loss_fn(policy, initial_state, ref_state, key, horizon_length):
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
    
    

    
    @eqx.filter_value_and_grad
    def compute_loss(policy, initial_state, horizon_length):
        mse_loss, _, _, _ = loss_fn(policy, initial_state, ref_state, horizon_length)
        return mse_loss

    
    
    @eqx.filter_jit
    def update_state(policy, initial_state, opt_state, horizon_length):
        
        loss, grads = compute_loss(policy, initial_state, horizon_length)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        policy = eqx.apply_updates(policy, updates)
        
        return loss, policy, opt_state

    
    
    
    if train_dpc_params.optimizer_type == 'adam':
        optimizer = optax.adam(train_dpc_params.learning_rate)
    elif train_dpc_params.optimizer_type == 'rmsprop':
        optimizer = optax.rmsprop(train_dpc_params.learning_rate, decay=0.9, eps=1e-8)
    elif train_dpc_params.optimizer_type == 'sgd':
        optimizer = optax.sgd(train_dpc_params.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {dpc_params.optimizer_type}")
    
    
    
    env = excenvs.make(env_params.name, batch_size=env_params.batch_size, l=env_params.l, m=env_params.m, tau=env_params.tau, max_torque=env_params.max_torque)
    
    ref_state = jnp.tile(jnp.array([[0, 0]]), (env_params.batch_size, 1)).astype(jnp.float32)
    
    
    #key, subkey = jax.random.split(key)
    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    
    
    policy = PolicyNetwork(train_dpc_params.layer_sizes, key=key)
    
    opt_state = optimizer.init(policy)

    losses = []
    
    start_time = time.time()

    
    
    
    
    for epoch in range(train_dpc_params.epochs):
        
        jax_key, subkey = jax.random.split(jax_key)
        
        
        if epoch < 5000:
            min_angle = 120
            horizon_length = train_dpc_params.horizon_lengths[0]
            
        elif 5000 < epoch:
            min_angle = 179
            horizon_length = train_dpc_params.horizon_lengths[1]
        
        
        train_initial_states = train_random_initial_state(subkey, min_angle, env_params.batch_size)
        
        loss, policy, opt_state = update_state(policy, train_initial_states, opt_state, horizon_length)
        
        losses.append(loss.item())
        
        _, predicted_states, actions, initial_states = loss_fn(policy, train_initial_states, ref_state, horizon_length)
        
        
        
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")
            
        

    
    total_training_time = time.time() - start_time
    print(f"Training time : {total_training_time:.2f} seconds")
    
    training_loss = jnp.array(losses)

    train_results = TrainingResults(training_loss=training_loss, state_trajectories=predicted_states, action_trajectories=actions)
    
    return policy , train_results
    
    





