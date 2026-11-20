"""JAX in double precision, as the engine runs: `chc.allocation` refuses to plan in single, and a
response built in single keeps its rounded parameters."""

import jax

jax.config.update("jax_enable_x64", True)
