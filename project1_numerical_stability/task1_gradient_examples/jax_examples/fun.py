from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

def g(x):
    return jnp.array([
        x[0]**5 + x[1]**3 * x[1]**2 + x[2],
        x[0]*x[1]*x[2],
        x[0] + x[1] + x[2]
    ])

x = jnp.array([2.0,2.0,2.0])

print("x = ",x)
print("g(x) = ",g(x))


print("fwd ")
print(jax.jacfwd(g)(x))

print("bck")
print(jax.jacrev(g)(x))


