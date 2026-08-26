# PI²-CMA-ES in JAX

A from-scratch JAX implementation of **Path Integral Policy Improvement with Covariance Matrix Adaptation** (PI²-CMA-ES), applied to neural-network policies on [gymnax](https://github.com/RobertTLange/gymnax) control environments.

> **Status:** coursework, complete for its original scope and not maintained. Kept public as a work sample. See *Scope and attribution* below, this was a group project and only my part is kept.

## What it is

PI²-CMA-ES is a **derivative-free** policy search method: rather than backpropagating through the environment, it samples policy parameters from a Gaussian, rolls out each sample, reweights by cost, updating both mean and covariance at each step based on a population average weigthed by individual performance.

The reference is Stulp & Sigaud, *Path Integral Policy Improvement with Covariance Matrix Adaptation*, ICML 2012. Equation numbers in comments refer to the paper's equations.

## Implementation Details

The whole optimization loop compiles into a single XLA program:

- The policy parameters are a flax pytree, flattened with `ravel_pytree` into the vector the search actually operates on, and unflattened for evaluation.
- Each episode is a `lax.scan` over environment steps, rollouts/batching is done through vmap
- The outer optimization is itself a `lax.scan` over generations, which means the entire run is a single compiled program.
- The covariance update is a reward-weighted outer product via `einsum`, with an exploration term added back each generation to keep the search from collapsing prematurely.
- Cost accumulation masks post-termination steps explicitly, so early-finishing episodes in a batch don't contaminate the return.

Environments: MountainCar (continuous output through `tanh`) and CartPole (discrete via `argmax`), with small MLP policies.

## Files

| File | What it is |
|---|---|
| `IPCMAES.py` | The PI²-CMA-ES algorithm — sampling, cost, reweighting, covariance update, the scanned outer loop. |
| `EnvSetup.py` | Policy networks and the jitted/vmapped gymnax rollout harness. |

## Scope and attribution

This was the final project for **CMPSCI 687 (Reinforcement Learning), UMass Amherst, Fall 2025**, done as a group. These files are my contribution. Other parts of the original project were written by teammates and are deliberately not included.

The `rollout` function in `EnvSetup.py` follows the pattern from the gymnax authors' tutorial on jit-wrapping environments; credit for that approach is theirs.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Jose Miguel Cruz — <jmcruz@umass.edu>
