from typing import Callable, Any

import jax
from flax import linen as nn
from functools import partial
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp


#Following a very useful tutorial provided by the gymnax authors on how to jit-wrap the environment
class MountainCarNet(nn.Module):
    """Simple NN for MountainCar."""

    num_hidden_units: int
    num_hidden_layers: int

    @nn.compact
    def __call__(self, x, key):
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return nn.tanh(x) #no gradients so will just straight up return the output velocity

class CartPoleNet(nn.Module):
    """Simple NN for CartPole."""

    num_hidden_units: int
    num_hidden_layers: int

    @nn.compact
    def __call__(self, x, key):
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
        x = nn.Dense(features=2)(x)
        return jnp.argmax(x, axis=-1)


@partial(jax.jit, static_argnames=("env", "env_params", "model", "steps_in_episode", "ravel_func"))
def rollout(key_input: jax.random.PRNGKey, env, env_params, model, policy_params: jax.Array, steps_in_episode: int, ravel_func: Callable[[jax.Array], Any]):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    policy_params = ravel_func(policy_params)
    key_reset, key_episode = jax.random.split(key_input)
    obs, state = env.reset(key_reset, env_params)

    def policy_step(state_input, tmp):
        """Step transition in jax env."""
        obs, state, policy_params, key = state_input
        key, key_step, key_net = jax.random.split(key, 3)
        action = model.apply(policy_params, obs, key_net)
        next_obs, next_state, reward, done, _ = env.step(
            key_step, state, action, env_params
        )
        carry = [next_obs, next_state, policy_params, key]
        return carry, [obs, action, reward, next_obs, done]

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step, [obs, state, policy_params, key_episode], (), steps_in_episode
    )
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done


vmapped_stepping = jax.vmap(rollout, in_axes=(0, None, None, None, 0, None, None))

@partial(jax.jit, static_argnames=("env", "env_params", "model", "ravel_fn", "num_trajectories"))
def sample_trajectories(key, env, env_params, model, flat_params, cov, num_trajectories, ravel_fn):
    """Handles the perturbation of parameters + calculating rewards"""
    key, parameter_key = jax.random.split(key)
    test_parameters = jax.random.multivariate_normal(
        parameter_key, mean=flat_params, cov=cov, shape=(num_trajectories,)
    )
    key_rollout = jax.random.split(key, num_trajectories)
    MAX_STEPS = env_params.max_steps_in_episode
    obs, action, reward, next_obs, done = vmapped_stepping(
        key_rollout, env, env_params, model, test_parameters, MAX_STEPS, ravel_fn
    )
    return obs, action, reward, done, test_parameters
