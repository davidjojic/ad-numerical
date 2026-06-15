from jax import config
config.update("jax_enable_x64",True)

import jax
import jax.numpy as jnp

def g(x):
    return jnp.array([
        x[0]*x[1],
        x[0]**2+jnp.sin(x[1])
    ])

x = jnp.array([2.0,0.5])

print("x = ",x)
print("g(x) = ",g(x))

print("jacfwd = ")
print(jax.jacfwd(g)(x))

print("jacrev =  ")
print(jax.jacrev(g)(x))
