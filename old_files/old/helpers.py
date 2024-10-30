import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


#random initial states for train

def train_random_initial_state(key, min_angle,  batch_size):
    
    theta_min_pos_rad = np.deg2rad(min_angle)
    theta_max_pos_rad = np.deg2rad(1)
    theta_min_neg_rad = np.deg2rad(-min_angle)
    theta_max_neg_rad = np.deg2rad(-1)
    
    
    keys = jax.random.split(key, batch_size )
    
    
    thetas_pos = jax.vmap(lambda k: jax.random.uniform(k, minval=theta_min_pos_rad, maxval=theta_max_pos_rad))(keys[:int(batch_size/2)])
    thetas_neg = jax.vmap(lambda k: jax.random.uniform(k, minval=theta_min_neg_rad, maxval=theta_max_neg_rad))(keys[int(batch_size/2):])
    
   
    thetas = jnp.concatenate([thetas_pos, thetas_neg])
    thetas = jax.random.permutation(keys[0], thetas)
    
   
    thetas = (thetas + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    
    return jnp.stack([thetas / jnp.pi, jnp.zeros_like(thetas)], axis=1)


#random initial states for test

def test_random_initial_state(key, batch_size):
    
    theta_min_pos_rad = np.deg2rad(175)
    theta_max_pos_rad = np.deg2rad(160)
    theta_min_neg_rad = np.deg2rad(-175)
    theta_max_neg_rad = np.deg2rad(-160)
    
    
    keys = jax.random.split(key, batch_size )
    
    
    thetas_pos = jax.vmap(lambda k: jax.random.uniform(k, minval=theta_min_pos_rad, maxval=theta_max_pos_rad))(keys[:int(batch_size/2)])
    thetas_neg = jax.vmap(lambda k: jax.random.uniform(k, minval=theta_min_neg_rad, maxval=theta_max_neg_rad))(keys[int(batch_size/2):])
    
    
    thetas = jnp.concatenate([thetas_pos, thetas_neg])
    thetas = jax.random.permutation(keys[0], thetas)
    
    thetas = (thetas + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    return jnp.stack([thetas / jnp.pi, jnp.zeros_like(thetas)], axis=1) 


#plot training  loss 

def plot_training_loss(train_results):
    
    training_loss = train_results.training_loss
    
    plt.plot(jnp.log(training_loss))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.show()



#plot training results 

def plot_train_results(train_results, tau):
    
    state_trajectories = train_results.state_trajectories
    action_trajectories = train_results.action_trajectories

    
    num_batches = state_trajectories.shape[1]  
    num_timesteps = state_trajectories.shape[0] 

    t = jnp.linspace(0, num_timesteps - 1, num_timesteps) * tau

    batch_idx = min(10, num_batches) 
    
    fig, axs = plt.subplots(nrows=batch_idx, ncols=1, figsize=(10, 4 * batch_idx), sharex=True)
    
    for i in range(batch_idx):
        
        axs[i].plot(t, state_trajectories[:, i, 0], label=r'$\theta$ (Observed)', color='b')
        axs[i].plot(t, state_trajectories[:, i, 1], label=r'$\omega$ (Observed)', color='g')
            
            
        axs[i].plot(t, action_trajectories[:, i], 'r--', label=r'$\theta$ (Action)')
        
        
        axs[i].set_title(f"Train Batch {i+1}")
        axs[i].set_ylabel("State")
        axs[i].legend(loc='upper right')
        axs[i].grid()

    
    axs[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.grid(True)
    plt.show()


    
#plot test results 

def plot_test_results(test_results, tau):
    
    test_state_trajectories = test_results.test_state_trajectories
    test_action_trajectories = test_results.test_action_trajectories

    
    num_batches = test_state_trajectories.shape[1]  
    num_timesteps = test_action_trajectories.shape[0] 

    #t = jnp.linspace(0, num_timesteps - 1, num_timesteps) * tau
    t = jnp.arange(num_timesteps) * tau

    batch_idx = min(10, num_batches) 
    
    fig, axs = plt.subplots(nrows=batch_idx, ncols=1, figsize=(10, 4 * batch_idx), sharex=True)
    
    for i in range(batch_idx):
        
        axs[i].plot(t, test_state_trajectories[:, i, 0], label=r'$\theta$ (Observed)', color='b')
        axs[i].plot(t, test_state_trajectories[:, i, 1], label=r'$\omega$ (Observed)', color='g')
            
            
        axs[i].plot(t, test_action_trajectories[:, i], 'r--', label=r'$\theta$ (Action)')
        
        
        axs[i].set_title(f"Test Batch {i+1}")
        axs[i].set_ylabel("State")
        axs[i].legend(loc='upper right')
        axs[i].grid()

    
    axs[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
