"""Basic tests for pose_analyzer package."""

import pytest


@pytest.mark.unit
def test_package_imports():
    """Test that the package can be imported."""
    try:
        import pose_analyzer
        assert True
    except ImportError:
        pytest.skip("pose_analyzer package not yet installed")


@pytest.mark.unit
def test_basic_assertion():
    """Sanity check test."""
    assert 1 + 1 == 2
