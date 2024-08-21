import numpy as np
from array_api_compat import get_namespace

from utils import array_api_compatible


def normalize(arr):
    xp = get_namespace(arr)
    mean = xp.mean(arr)
    std = xp.std(arr)
    normalized_arr = (arr - mean) / std

    return normalized_arr


@array_api_compatible
def test_normalize(xp, device):
    arr = xp.asarray([1., 2., 3., 4., 5., 6.], device=device, dtype=xp.float32)
    res = normalize(arr)

    assert abs(xp.mean(res)) < 1e-16
    assert abs(xp.std(res) - 1) < 1e16
