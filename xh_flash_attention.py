import numpy as np
import jax.numpy as jnp
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import functools
from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention


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



def tpu_flash_attention_lanuch(
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
    # axis_names = nn.logical_to_mesh_axes(self.flash_axis_names)
    axis_names = jax.sharding.PartitionSpec("dp", "head", "fsdp", "mp")
    segment_axis_names = jax.sharding.PartitionSpec(
        ("dp", 'activation_length_no_heads')
    )
    # segment_axis_names = nn.logical_to_mesh_axes(
    #     (BATCH, 'activation_length_no_heads')
    # )

    mesh = get_jax_mesh(('1,  1,  -1,  1'), ('dp', 'head', 'fsdp', 'mp'))

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





