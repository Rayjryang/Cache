
import flax.linen as nn
import flax
import jax
import jax.numpy as jnp
from jax.numpy import einsum
from einops import rearrange
import time
# from xh_flash_attention import tpu_flash_attention_lanuch
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import functools
from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention
import numpy as np
from jax import make_jaxpr





def get_jax_mesh(axis_dims, names):
    if axis_dims.startswith('!'):
        # Allow splitting a physical mesh axis if needed
        mesh_axis_splitting = True
        axis_dims = axis_dims[1:]
    else:
        mesh_axis_splitting = False

    if ':' in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(','):
            name, dim = axis.split(':')
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)
        assert(set(dim_names) == set(names))
    else:
        dims = [int(x) for x in axis_dims.split(',')]
        dim_names = names
    assert len(dims) == len(names)
    mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
    # print(f"mesh_shape: {mesh_shape}") #(8, 1, 1, 1)
    if mesh_axis_splitting:
        physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
    else:
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    # print(f'physical_mesh: {physical_mesh}. dim_names: {dim_names}')
    return Mesh(physical_mesh, dim_names)

#mesh = get_jax_mesh(('-1, 1, 1, 1'), ('dp', 'head', 'fsdp', 'mp'))
#mesh = get_jax_mesh(('1, 1, -1, 1'), ('dp', 'head', 'fsdp', 'mp'))
#mesh = get_jax_mesh(('4, 1, 2, 1'), ('dp', 'head', 'fsdp', 'mp'))
#mesh = get_jax_mesh(('8, 1, 1, 1'), ('dp', 'head', 'fsdp', 'mp'))


mesh = get_jax_mesh(('-1, 1, 1, 1'), ('dp', 'head', 'fsdp', 'mp'))
# print(f"mesh devices {mesh.devices}")
# mesh = get_jax_mesh(('1, 1, -1, 1'), ('dp', 'head', 'fsdp', 'mp'))
axis_names = jax.sharding.PartitionSpec("dp", "head", "fsdp", "mp")
segment_axis_names = jax.sharding.PartitionSpec(
    ("dp", 'activation_length_no_heads'),
    #("fsdp", 'activation_length_no_heads')
)

# axis_names = jax.sharding.PartitionSpec("dp"*"head", "fsdp"*"mp")


@functools.partial(
    shard_map,
    mesh=mesh,
    in_specs=(
        axis_names,
        axis_names,
        axis_names,
    ),
    out_specs=axis_names,
    check_rep=False,
)
def wrap_flash_attention(query, key, value, decoder_segment_ids = None):
    if decoder_segment_ids is not None:
        assert (
            query.shape[2]
            == decoder_segment_ids.q.shape[1]
        ), 'Sharding along sequence dimension not allowed in flash attention'
    
  
    return tpu_flash_attention.flash_attention(
        query,
        key,
        value,
        causal=False,
        segment_ids=decoder_segment_ids,
        sm_scale = 64 ** -0.5,
        block_sizes=tpu_flash_attention.BlockSizes(
            block_q=min(512, query.shape[2]),
            block_k_major=min(512, key.shape[2]),
            block_k=min(512, key.shape[2]),
            block_b=min(2, query.shape[0]),
            block_q_major_dkv=min(512, query.shape[2]),
            block_k_major_dkv=min(512, key.shape[2]),
            block_q_dkv=min(512, query.shape[2]),
            block_k_dkv=min(512, key.shape[2]),
            block_q_dq=min(1024, query.shape[2]),
            block_k_dq=min(256, key.shape[2]),
            block_k_major_dq=min(512, key.shape[2]),
        ),
    )



@functools.partial(
    shard_map,
    mesh=mesh,
    in_specs=(
        axis_names,
        axis_names,
        axis_names,
    ),
    out_specs=axis_names,
    check_rep=False,
)
def wrap_vanilla_attention(q, k, v, decoder_segment_ids = None):
    
    scale = 64 ** -0.5 
    dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale
    attn = nn.softmax(dots, axis = -1)
    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    # print(f"vanilla output shape: {out.shape}")
    # out = rearrange(out, 'b h n d -> b n (h d)')

    return  out



def get_numerial_difference(vanilla_output, flash_output):

    print(f'vanilla_output: {vanilla_output[0,0,:10]}')
    
    print(f'flash_output: {flash_output[0,0,:10]}')

    diff = jnp.abs(vanilla_output) - jnp.abs(flash_output)
    sum_diff = jnp.sum(abs(diff))
    print("sum of differences:", sum_diff, 'the avg difference for each element:',sum_diff/(jnp.prod(jnp.array(vanilla_output.shape))))


if __name__ == '__main__':
   
    
    run_n = 20


    dim = 768
    heads = 12
    dim_head = 64
    num_head = 12

    batch_size = 256
    seq_len = 256
    
    # preparing input
    key = jax.random.PRNGKey(0)
    # input = jax.random.normal(key, (batch_size, seq_len, dim))
    key, subkey = jax.random.split(key)
    
    q = jax.random.normal(key, (batch_size, num_head, seq_len, dim_head))
    k = jax.random.normal(key, (batch_size, num_head, seq_len, dim_head))
    v = jax.random.normal(key, (batch_size, num_head, seq_len, dim_head))


    # vanilla attention
    wrap_vanilla_attention(q,k,v).block_until_ready()


    start_time = time.time()
    for _ in range(run_n):
        vanilla_output = wrap_vanilla_attention(q,k,v).block_until_ready()
    end_time = time.time()
    vanilla_elapsed_time = end_time - start_time
    print(f"vanilla attention forward pass took {vanilla_elapsed_time} seconds")

    vanilla_output = rearrange(vanilla_output, 'b h n d -> b n (h d)')

    # flash attention
    start_time = time.time()
    for _ in range(run_n):
        flash_output =wrap_flash_attention(q,k,v).block_until_ready()
    end_time = time.time()
    flash_elapsed_time = end_time - start_time
    print(f"flash attention forward pass took {flash_elapsed_time} seconds")
    flash_output = rearrange(flash_output, 'b h n d -> b n (h d)')



    # comparison
    print(f"batch_size {batch_size}, seq_len: {seq_len}, speed {vanilla_elapsed_time/flash_elapsed_time:.2f}. times")
    
    get_numerial_difference(vanilla_output, flash_output)
