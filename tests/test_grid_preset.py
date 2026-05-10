"""Grid preset sizes for tuning (pure combinatorics)."""

from metaorder_signal.empirical.tuning import grid_preset_size


def test_fine_grid_is_larger_than_fast():
    assert grid_preset_size("fine") > grid_preset_size("fast")


def test_preset_counts_product():
    assert grid_preset_size("fast") == 48
    assert grid_preset_size("fine") == 128
