
import flax.linen as nn
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



class Attention(nn.Module):
    dim: int
    heads: int = 8
    dim_head: int = 64
    attention_kernel: str = None
    attention_config: dict = None

    
    def tpu_flash_attention(
        self,
        query,
        key,
        value,
        decoder_segment_ids: None):
        """TPU Flash Attention."""
        # Transpose to ('batch', 'heads', 'length', 'kv')
        # query = jnp.transpose(query, axes=(0, 2, 1, 3))
        # key = jnp.transpose(key, axes=(0, 2, 1, 3))
        # value = jnp.transpose(value, axes=(0, 2, 1, 3))
        if not(query.shape[1] == key.shape[1] == value.shape[1]):
            raise ValueError(f"The flash attention kernel requires Q, K and V to have the same number of heads"
                            "{query.shape=} {key.shape=}, {value.shape=}")

        if decoder_segment_ids is not None:
            decoder_segment_ids = tpu_flash_attention.SegmentIds(
                decoder_segment_ids, decoder_segment_ids
            )

        axis_names = jax.sharding.PartitionSpec("dp", "head", "fsdp", "mp")
        segment_axis_names = jax.sharding.PartitionSpec(
            ("dp", 'activation_length_no_heads')
        )

        # axis_names = nn.logical_to_mesh_axes(("dp", "head", "fsdp", "mp"))
        # segment_axis_names = nn.logical_to_mesh_axes(
        #     ('dp', 'activation_length_no_heads')
        # )


        # mesh = get_jax_mesh(('1,  1,  -1,  1'), ('dp', 'head', 'fsdp', 'mp'))
        mesh = get_jax_mesh(('-1, 1, 1, 1'), ('dp', 'head', 'fsdp', 'mp'))
        # print(f'mesh: {mesh}')

        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                axis_names,
                axis_names,
                axis_names,
                segment_axis_names,
            ),
            out_specs=axis_names,
            check_rep=False,
        )
        def wrap_flash_attention(query, key, value, decoder_segment_ids):
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

        devices_in_data_fsdp =  mesh.shape['dp'] * mesh.shape['fsdp']
        assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
            'Batch dimension should be shardable among the devices in data and fsdp'
            ' axis'
        )
        x = wrap_flash_attention(query, key, value, decoder_segment_ids)
        # x = jnp.transpose(x, axes=(0, 2, 1, 3))
        return x


    @nn.compact
    def __call__(self, x):
        
        inner_dim = self.dim_head * self.heads
        scale = self.dim_head ** -0.5
        to_qkv = nn.Dense(features = inner_dim * 3, use_bias = False)(x)
        qkv = jnp.split(to_qkv, 3, axis = -1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        assert self.attention_kernel in ['normal','flash_attention']
        # print(f"q[0,0,0,:10]: {q[0,0,0,:10]}") # check the input data. ensuring it is same for vanilla and flash attention

        if self.attention_kernel == 'normal':
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale
            attn = nn.softmax(dots, axis = -1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            # print(f"vanilla output shape: {out.shape}")
            out = rearrange(out, 'b h n d -> b n (h d)')
           
        else:
            out = self.tpu_flash_attention(query = q, key = k, value = v, decoder_segment_ids=None)
            # print(f"flash output shape: {out.shape}")
            out = rearrange(out, 'b h n d -> b n (h d)')

        return out


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


    batch_size = 32
    seq_len = 1024
    
    # preparing input
    key = jax.random.PRNGKey(0)
    input = jax.random.normal(key, (batch_size, seq_len, dim))
    key, subkey = jax.random.split(key)


    # vanilla attention
    vanilla_attention = Attention(dim=dim, heads=heads, dim_head=dim_head, attention_kernel = 'normal', attention_config = None)
    params_vanilla = vanilla_attention.init(subkey, input)
    vanilla_attention.apply(params_vanilla, input).block_until_ready()


    start_time = time.time()
    for _ in range(run_n):
        vanilla_output = vanilla_attention.apply(params_vanilla, input).block_until_ready()
    end_time = time.time()
    vanilla_elapsed_time = end_time - start_time
    print(f"vanilla attention forward pass took {vanilla_elapsed_time} seconds")


    # flash attention
    flash_attention = Attention(dim=dim, heads=heads, dim_head=dim_head, attention_kernel = 'flash_attention', attention_config = None)
    params_flash = flash_attention.init(subkey, input)
    flash_attention.apply(params_flash, input).block_until_ready()


    start_time = time.time()
    for _ in range(run_n):
        flash_output = flash_attention.apply(params_flash, input).block_until_ready()
    end_time = time.time()
    flash_elapsed_time = end_time - start_time
    print(f"flash attention forward pass took {flash_elapsed_time} seconds")


    # comparison
    print(f"batch_size {batch_size}, seq_len: {seq_len}, speed {vanilla_elapsed_time/flash_elapsed_time:.2f}. times")
    
    get_numerial_difference(vanilla_output, flash_output)


'''output:

vanilla attention forward pass took 1.334207534790039 seconds

flash attention forward pass took 17.17262029647827 seconds

batch_size 32, seq_len: 1024, speed 0.08. times

vanilla_output: [ 0.01298839 -0.02802607 -0.00647758  0.00275855 -0.0326768  -0.10131085
  0.00496649  0.13378385  0.03150739  0.00630885]

flash_output: [ 0.01291609 -0.02798277 -0.00648307  0.00279816 -0.03274597 -0.10131068
  0.00481233  0.13376313  0.03156951  0.00636329]
  
sum of differences: 2321.9592 the avg difference for each element: 9.2266375e-05

'''