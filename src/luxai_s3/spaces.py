from gymnax.environments.spaces import Space
import jax.numpy as jnp
import chex
import jax
import numpy as np
class MultiDiscrete(Space):
    """Minimal jittable class for multi discrete gymnax spaces."""

    def __init__(self, nvec: np.ndarray):
        self.nvec = nvec
        self.n = nvec.shape[0]
        self.shape = nvec
        self.dtype = jnp.int16

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        return (jax.random.uniform(rng, shape=(self.n, ), minval=0, maxval=1) * self.nvec).astype(self.dtype)

    def contains(self, x: jnp.int_) -> jnp.ndarray:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond