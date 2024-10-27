import jax

#from utils.dpc_controller import EnvParameters , TrainDPCParameters , train_dpc_controller , loss_fn
from utils.old.dpc_controller import EnvParameters , TrainDPCParameters , train_dpc_controller 
from utils.old.test_dpc import TestDPCParameters , test_dpc_controller
from utils.old.helpers import plot_train_results , plot_training_loss , plot_test_results



def main():
    key = jax.random.PRNGKey(0)
    key_test = jax.random.PRNGKey(1)
    
    env_params = EnvParameters(
        name='Pendulum-v0',
        batch_size=16,
        l=1,
        m=1,
        tau=1e-2,
        max_torque=2
    )
    
    train_dpc_params = TrainDPCParameters(
        layer_sizes=[2, 64, 1],
        learning_rate=1e-3,
        optimizer_type='adam',
        epochs=20_000,
        horizon_lengths=[300, 500]
    )
    
    
    test_dpc_params = TestDPCParameters(
        #layer_sizes=[4, 20, 1],
        horizon_length=1000
    )
    
    
    policy, train_results = train_dpc_controller(env_params, train_dpc_params, key)
    #plot_train_results(train_results, env_params.tau)
    plot_training_loss(train_results)
    test_results = test_dpc_controller(policy, env_params, train_dpc_params, test_dpc_params, key_test)
    plot_test_results(test_results, env_params.tau)

if __name__ == "__main__":
    main()
