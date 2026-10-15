import pytest

from egonets_wl.kernels.wl_subtree import WLSubtreeKernel


def test_wl_kernel_fails_explicitly_until_implemented():
    with pytest.raises(NotImplementedError):
        WLSubtreeKernel().fit_transform([])
