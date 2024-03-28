
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.numpy import einsum
from einops import rearrange
from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention
import time



class Attention(nn.Module):
    dim: int
    heads: int = 8
    dim_head: int = 64
    attention_kernel: str = None
    attention_config: dict = None

    def tpu_flash_attention_wrapper(
        self,
        query,
        key,
        value):
        """TPU Flash Attention."""
        #input:  ('batch', 'heads', 'length', 'kv')

        x = tpu_flash_attention.flash_attention(
                query,
                key,
                value,
                causal=False,
                segment_ids=None,
                sm_scale = self.dim_head ** -0.5,
                block_sizes=tpu_flash_attention.BlockSizes(
                    block_q=min(self.attention_config['block_q'], query.shape[2]),
                    block_k_major=min(self.attention_config['block_k_major'], key.shape[2]),
                    block_k=min(self.attention_config['block_k'], key.shape[2]),
                    block_b=min(self.attention_config['block_b'], query.shape[0]),
                    block_q_major_dkv=min(self.attention_config['block_q_major_dkv'], query.shape[2]),
                    block_k_major_dkv=min(self.attention_config['block_k_major_dkv'], key.shape[2]),
                    block_q_dkv=min(self.attention_config['block_q_dkv'], query.shape[2]),
                    block_k_dkv=min(self.attention_config['block_k_dkv'], key.shape[2]),
                    block_q_dq=min(self.attention_config['block_q_dq'], query.shape[2]),
                    block_k_dq=min(self.attention_config['block_k_dq'], key.shape[2]),
                    block_k_major_dq=min(self.attention_config['block_k_major_dq'], key.shape[2]),
                ),
                debug = False,
            )
    
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
        print(f"q[0,0,0,:10]: {q[0,0,0,:10]}") # check the input data. ensuring it is same for vanilla and flash attention

        if self.attention_kernel == 'normal':
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale
            attn = nn.softmax(dots, axis = -1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            print(f"vanilla output shape: {out.shape}")
            out = rearrange(out, 'b h n d -> b n (h d)')
           
        else:
            out = self.tpu_flash_attention_wrapper(q, k, v)
            print(f"flash output shape: {out.shape}")
            out = rearrange(out, 'b h n d -> b n (h d)')

        return out


def get_numerial_difference(vanilla_output, flash_output):

    print(f'vanilla_output: {vanilla_output[0,0,:10]}')
    
    print(f'flash_output: {flash_output[0,0,:10]}')

    diff = abs(vanilla_output) - abs(flash_output)
    sum_diff = jnp.sum(abs(diff))
    print("sum of differences:", sum_diff)


if __name__ == '__main__':
   
    
    block_size = 128
    attention_config_flash_attention = {
                'block_q': block_size,
                'block_k_major': block_size,
                'block_k': block_size,
                'block_b': 2,
                'block_q_major_dkv': block_size,
                'block_k_major_dkv': block_size,
                'block_q_dkv': block_size,
                'block_k_dkv': block_size,
                'block_q_dq': block_size,
                'block_k_dq': block_size,
                'block_k_major_dq': block_size,
            }

    dim = 768
    heads = 12
    dim_head = 64


    batch_size = 128
    seq_len = 512
   
    # preparing input
    key = jax.random.PRNGKey(0)
    input = jax.random.normal(key, (batch_size, seq_len, dim))
    key, subkey = jax.random.split(key)


    # vanilla attention
    vanilla_attention = Attention(dim=dim, heads=heads, dim_head=dim_head, attention_kernel = 'normal', attention_config = None)
    params_vanilla = vanilla_attention.init(subkey, input)

    start_time = time.time()
    vanilla_output = vanilla_attention.apply(params_vanilla, input)
    end_time = time.time()
    vanilla_elapsed_time = end_time - start_time
    print(f"vanilla attention forward pass took {vanilla_elapsed_time} seconds")


    # flash attention
    flash_attention = Attention(dim=dim, heads=heads, dim_head=dim_head, attention_kernel = 'flash_attention', attention_config = attention_config_flash_attention)
    params_flash = flash_attention.init(subkey, input)

    start_time = time.time()
    flash_output = flash_attention.apply(params_flash, input)
    end_time = time.time()
    flash_elapsed_time = end_time - start_time
    print(f"flash attention forward pass took {flash_elapsed_time} seconds")


    # comparison
    print(f"seq_len: {seq_len}. speed {vanilla_elapsed_time/flash_elapsed_time:.2f}. times")
    
    get_numerial_difference(vanilla_output, flash_output)

