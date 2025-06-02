import numpy as np


try:
    import torch
except ImportError:
    torch = None


def maybe_expand_dims(env, x):
    if hasattr(env, "batch_size"):
        return np.expand_dims(x, axis=0)  # Expand dims by 1 if batch_size exists
    return x


# https://github.com/pytorch/pytorch/issues/52161
def has_integer_dtype(tensor, signed: bool | None = None) -> bool:
    """Determines if a PyTorch tensor has an integer dtype.

    It also can force `tensor` to be signed or unsigned.

    Parameters
    ----------
    tensor
        The tensor to check.
    signed
        Determines which dtypes are allowed for `tensor`:

        - If ``None`` both unsigned and signed integer will be allowed.

        - If ``False`` only unsigned dtypes will be allowed.

        - If ``True`` only signed dtypes will be allowed.

    Returns
    -------
    bool
        ``True`` if the input tensor satisfies the requested condition, ``False``
        otherwise.

    """
    assert torch is not None

    uint_types = [torch.uint8]
    sint_types = [torch.int8, torch.int16, torch.int32, torch.int64]
    if signed is None:
        return tensor.dtype in uint_types + sint_types
    elif signed:
        return tensor.dtype in sint_types
    else:
        return tensor.dtype in uint_types
