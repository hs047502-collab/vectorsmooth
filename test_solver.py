"""Regression tests for the Schneider cubic solver.

The residual was once ``pts - (P0*B0 + P3*B3)`` (missing the B1/B2
weights), which gave err ~27.5 on a perfect quarter arc and silently
disabled every cubic fit. These tests pin the fixed behaviour.
"""
import numpy as np

from core.geometry import fit_single_cubic


def _quarter_arc(r=70.0, n=64):
    t = np.linspace(0, np.pi / 2, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _wiggly_line(n=64, amp=1.0, waves=3.0):
    x = np.linspace(0, 100, n)
    y = amp * np.sin(2 * np.pi * waves * x / 100.0)
    return np.stack([x, y], axis=1)


def test_quarter_arc_fits_cubic():
    (_, _, _, _), err, _ = fit_single_cubic(_quarter_arc())
    assert err < 1.0, 'quarter-arc cubic err %.3f (solver broken?)' % err


def test_gentle_wave_fits_cubic():
    (_, _, _, _), err, _ = fit_single_cubic(_wiggly_line(waves=0.5))
    assert err < 1.0, 'gentle-wave cubic err %.3f (solver broken?)' % err


def test_multiwave_line_does_not_blow_up():
    # 3 sine waves have real inflections no single cubic can follow;
    # err ~3 is legitimate, err ~29.6 was the broken-solver signature.
    (_, _, _, _), err, _ = fit_single_cubic(_wiggly_line())
    assert err < 6.0, 'wiggly-line cubic err %.3f (solver broken?)' % err


def test_straight_line_near_zero_err():
    pts = np.stack([np.linspace(0, 100, 32), np.zeros(32)], axis=1)
    (_, _, _, _), err, _ = fit_single_cubic(pts)
    assert err < 1e-6, 'straight-line cubic err %.3e' % err
