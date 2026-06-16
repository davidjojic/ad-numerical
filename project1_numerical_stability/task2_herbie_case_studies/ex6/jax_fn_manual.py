import jax
import jax.numpy as jnp

def fn(x):
    t0 = jnp.array([0.3568308184690063, 0.9356711744085836, 0.743076798396197, 0.3434118421253949, 0.29743319771194776], dtype=x.dtype)
    t1 = jnp.array([0.16033089003536888, 0.1905743829268704, 0.31940949525993967, 0.21856183912380575, 0.9376026929080123], dtype=x.dtype)
    s0 = jnp.sinh(x[2])
    s1 = ((jnp.arctan2(x[4], s0)) + ((s0 * s0) + jnp.abs(s0)))
    s2 = (jnp.arctan2(((jnp.arctan2(s1, x[1])) - (jnp.arctan2(x[1], x[1]))), jnp.sin(jnp.sin(s0))))
    t0 = ((jnp.exp((t0 - s1)) / x) + x)
    t1 = jnp.power(2.0 * jnp.arctan(jnp.exp(-t0)), x) + t0
    out = (((jnp.exp(s1) * jnp.exp(x)) + jnp.arcsinh(t1)) + t1)
    return out