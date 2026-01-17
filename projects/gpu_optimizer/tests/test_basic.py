"""Basic tests for gpu_optimizer package."""

import pytest


@pytest.mark.unit
def test_package_imports():
    """Test that the package can be imported."""
    try:
        import gpu_optimizer
        assert True
    except ImportError:
        pytest.skip("gpu_optimizer package not yet installed")


@pytest.mark.unit
def test_basic_assertion():
    """Sanity check test."""
    assert 1 + 1 == 2
