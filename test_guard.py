"""Tests for the accept-or-rollback guard (the core promise)."""
import numpy as np

from core.geometry import guard_candidate


def _square(n=25, s=100.0):
    side = np.linspace(0, s, n)
    top = np.stack([side, np.zeros(n)], axis=1)
    right = np.stack([np.full(n - 1, s), side[1:]], axis=1)
    bottom = np.stack([side[-2::-1], np.full(n - 1, s)], axis=1)
    left = np.stack([np.zeros(n - 2), side[-2:0:-1]], axis=1)
    return np.concatenate([top, right, bottom, left])


def test_identical_passes():
    sq = _square()
    ok, reason, dev = guard_candidate(sq, sq.copy(), 1.0, True)
    assert ok, reason
    assert dev == 0.0


def test_over_budget_rejected():
    sq = _square()
    shifted = sq + np.array([10.0, 0.0])
    ok, reason, dev = guard_candidate(sq, shifted, 1.0, True)
    assert not ok and 'budget' in reason, (ok, reason)
    assert dev > 9.0


def test_self_intersection_rejected():
    sq = _square()
    bowtie = np.array([[0., 0.], [100., 100.], [100., 0.], [0., 100.]])
    ok, reason, _ = guard_candidate(sq, bowtie, 500.0, True)
    assert not ok and 'self-intersection' in reason, (ok, reason)


def test_winding_flip_rejected():
    sq = _square()          # clockwise in y-down?  orientation is consistent
    flipped = sq[::-1].copy()
    from core.geometry import signed_area
    ok, reason, _ = guard_candidate(sq, flipped, 500.0, True,
                                    area_orig=signed_area(sq))
    assert not ok and 'winding' in reason, (ok, reason)
