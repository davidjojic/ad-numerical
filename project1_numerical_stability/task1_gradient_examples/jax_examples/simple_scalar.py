from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp


def f(x):
    return x[0]**2 + jnp.sin(x[1]) + x[0] * x[1]


x = jnp.array([2.0, 0.5])

print("x =", x)
print("f(x) =", f(x))

print("grad using jax.grad =", jax.grad(f)(x))
print("jacobian using jacfwd =", jax.jacfwd(f)(x))
print("jacobian using jacrev =", jax.jacrev(f)(x))
print("hessian =\n", jax.hessian(f)(x))