import array_api_compat
import array_api_strict
import numpy as np
import pytest


# We store all available backends for testing.  This could be build as here
# or be configured e.g. via environment variables.
# scipy and scikit-learn both have similar but different setups.
available_backends = {
    "numpy": (np, "cpu"),
    "strict": (array_api_strict, "cpu"),
}

try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    pass
else:
    # Torch was found, so we can use it:
    available_backends["pytorch[cpu]"] = (torch, "cpu")
    if torch.cuda.is_available():
        available_backends["pytorch[cuda]"] = (torch, "cuda")

try:
    import cupy  # type: ignore[import-not-found]
except ImportError:
    pass
else:
    # Torch was found, so we can use it:
    available_backends["cupy"] = (cupy, "cuda")


def get_available_backends():
    for key, (array_mod, device) in available_backends:
        # The module is the original module, use array_api_compat to fetch
        # the compatible namespace:
        xp = array_api_compat.get_namespace(array_mod.asarray(1))
        yield pytest.param(xp, device, name=key)

array_api_compatible = pytest.mark.parametrize("xp, device, to_numpy")
