import numpy as np

from metaorder_signal.losses import composite_loss, loss_sql_r2


def test_perfect_line_has_zero_sql_loss():
    x = np.log(np.linspace(1.0, 100.0, 50))
    y = 0.5 * x + 1.0
    ell = loss_sql_r2(x, y)
    assert ell < 1e-9


def test_composite_weights():
    v = composite_loss(1.0, 1.0, 1.0, 1.0, weights=[1.0, 0.0, 0.0, 0.0])
    assert np.isclose(v, 1.0)
