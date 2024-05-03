from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map


devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('i', 'j'))

a = jnp.arange( 8 * 16.).reshape(8, 16)
b = jnp.arange(16 * 32.).reshape(16, 32)

@partial(shard_map, mesh=mesh, in_specs=(P('i', 'j'), P('j', None)),
         out_specs=P('i', None))
def matmul_basic(a_block, b_block):
  # a_block: f32[2, 8]
  # b_block: f32[8, 32]
  z_partialsum = jnp.dot(a_block, b_block)
  z_block = jax.lax.psum(z_partialsum, 'j')
  return z_block

c = matmul_basic(a, b)  # c: f32[8, 32]ååååååå
#print(f'c in shard: {c}')


vanilla_c = jnp.dot(a, b)
#print(f'vanilla_c in shard: {vanilla_c}')

# Check if the matrices are equal
are_equal = jnp.allclose(c, vanilla_c)

print(are_equal)  # Thi
