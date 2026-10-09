"""Fused prompt attention for the native Windows H100 build."""
from contextlib import contextmanager


@contextmanager
def h100_attention():
    import torch.nn.functional as functional
    from torch.nn.attention import sdpa_kernel, SDPBackend
    original = functional.scaled_dot_product_attention

    def attention(query, key, value, *args, **kwargs):
        if query.device.type != 'cuda':
            return original(query, key, value, *args, **kwargs)
        backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if query.shape[-2] == 1:
            backends.append(SDPBackend.MATH)
        with sdpa_kernel(backends, set_priority=True):
            return original(query, key, value, *args, **kwargs)

    functional.scaled_dot_product_attention = attention
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original
