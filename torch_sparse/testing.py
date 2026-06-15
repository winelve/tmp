from typing import Any

import pytest
import torch
import torch_scatter
from packaging import version

reductions = ['sum', 'add', 'mean', 'min', 'max']

dtypes = [torch.half, torch.float, torch.double, torch.int, torch.long]
grad_dtypes = [torch.half, torch.float, torch.double]

if version.parse(torch_scatter.__version__) > version.parse("2.0.9"):
    dtypes.append(torch.bfloat16)
    grad_dtypes.append(torch.bfloat16)

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda:0')]
if hasattr(torch, 'sdaa') and torch.sdaa.is_available():
    devices += [torch.device('sdaa')]

# SDAA currently only supports float/double/int32/int64
SDAA_SUPPORTED_DTYPES = {torch.float, torch.double, torch.int, torch.long}


def is_dtype_supported(dtype: torch.dtype, device: torch.device) -> bool:
    """Check if dtype is supported on the given device."""
    if device.type == 'sdaa':
        return dtype in SDAA_SUPPORTED_DTYPES
    return True


def skip_if_unsupported(dtype: torch.dtype, device: torch.device):
    """Skip test if dtype is not supported on device."""
    if not is_dtype_supported(dtype, device):
        pytest.skip(f"dtype {dtype} not supported on {device}")


def tensor(x: Any, dtype: torch.dtype, device: torch.device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)
