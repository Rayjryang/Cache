import numpy as np
import jax.numpy as jnp
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import functools


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
    if mesh_axis_splitting:
        physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
    else:
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(physical_mesh, dim_names)



def tpu_flash_attention(
        mesh,
        query: Array,
        key: Array,
        value: Array,
        decoder_segment_ids = None) -> Array:
    """TPU Flash Attention."""
    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 1, 3))
    value = jnp.transpose(value, axes=(0, 2, 1, 3))

    if decoder_segment_ids is not None:
        decoder_segment_ids = splash_attention_kernel.SegmentIds(
            decoder_segment_ids, decoder_segment_ids
        )
    axis_names = jax.sharding.PartitionSpec("dp", "head", "fsdp", "mp")
    segment_axis_names = jax.sharding.PartitionSpec(
        ("dp", 'activation_length_no_heads')
    )

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
            ), 'Sharding along sequence dimension not allowed in tpu kernel attention'
        block_size=512
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=min(block_size, query.shape[2]),
            block_kv_compute=min(block_size, key.shape[2]),
            block_kv=min(block_size, key.shape[2]),
            block_q_dkv=min(block_size, query.shape[2]),
            block_kv_dkv=min(block_size, key.shape[2]),
            block_kv_dkv_compute=min(block_size, query.shape[2]),
            block_q_dq=min(block_size, query.shape[2]),
            block_kv_dq=min(block_size, query.shape[2]),
        )

        masks = [splash_attention_mask.CausalMask(shape=(query.shape[2], query.shape[2])) for i in
                 range(query.shape[1])]
        multi_head_mask = splash_attention_mask.MultiHeadMask(masks=masks)
        splash_kernel = splash_attention_kernel.make_splash_mha(mask=multi_head_mask,
                                                                head_shards=1,
                                                                q_seq_shards=1,
                                                                block_sizes=block_sizes)

        return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

    devices_in_data_fsdp = mesh.shape['dp'] * mesh.shape['fsdp']
    assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        'Batch dimension should be shardable among the devices in data and fsdp'
        ' axis'
    )
    x = wrap_flash_attention(query, key, value, decoder_segment_ids)
    x = jnp.transpose(x, axes=(0, 2, 1, 3))
    return x

mesh = get_jax_mesh(('1,  1,  -1,  1'), ('dp', 'head', 'fsdp', 'mp'))


