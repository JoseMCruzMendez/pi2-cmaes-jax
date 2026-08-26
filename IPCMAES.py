from jax import numpy as jnp
import jax
from functools import partial
from jax.flatten_util import ravel_pytree
from EnvSetup import sample_trajectories
from flax.linen import softmax

partial(jax.jit, static_argnames=("gamma",))
def cost_function(rewards, done, gamma):
    """Thin wrapper to make sure I don't miss the negative since CMAES minimizes + handle masking"""
    #This selects only entries that are 0 before any 1s can appear. It also correctly selects the first 1.
    valid_states = (done.astype(jnp.float32).cumsum(axis=-1) - done) == 0
    scaled_gamma = gamma ** jnp.arange(rewards.shape[-1])
    return - (rewards * scaled_gamma * valid_states).sum(axis=-1)

def exploration_cost(cov):
    return (cov * cov).sum()

@partial(jax.jit, static_argnames=["env", "env_params", "model", "num_steps", "gamma", "K"])
def IP2CMAES(key, env, env_params, model, model_params, num_steps=1_000, K=10., temp=1., initial_exploration=1e2, exploration_parameter=1e2, gamma=0.99):
    flat_params, _ = ravel_pytree(model_params)
    cov = jnp.eye(*flat_params.shape) * initial_exploration
    base_theta, ravel_fn = ravel_pytree(model_params)
    def IP2CMAES_step(carry, key):
        trajectory_key, parameter_key = jax.random.split(key)
        cur_params, cov = carry

        #Samples trajectories
        states, actions, rewards, done, test_parameters = sample_trajectories(
            trajectory_key, env, env_params, model, cur_params, cov=cov, num_trajectories=K, ravel_fn=ravel_fn
        )

        costs = cost_function(rewards, done, gamma=gamma) #(11)
        prob_weights = softmax(costs/(-temp)) #(13)
        new_params = (prob_weights[:, None] * test_parameters).sum(axis=0) #(15)
        centered_params = test_parameters - cur_params
        #Outer prod, so should go from [batch, params] to [batch, params, params]
        new_cov = (prob_weights[:, None, None] * jnp.einsum("bp,bk->bpk", centered_params, centered_params)).sum(axis=0)
        new_cov = new_cov + exploration_parameter * jnp.eye(*flat_params.shape)
        cur_cost = exploration_cost(cov)
        carry = (new_params, new_cov)
        return carry, (costs, cur_cost)
    scan_keys = jax.random.split(key, num_steps)
    (final_flat_params, final_cov), (all_costs, all_exp_costs) = jax.lax.scan(
        IP2CMAES_step, (flat_params, cov), scan_keys, num_steps
    )
    return ravel_fn(final_flat_params), all_costs, all_exp_costs
