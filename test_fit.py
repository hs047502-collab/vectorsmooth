"""Tests for the greedy L-vs-C span fitter."""
import numpy as np

from core.geometry import fit_span


def test_straight_run_collapses_to_one_L():
    x = np.linspace(0, 200, 200)
    rng = np.random.default_rng(7)
    pts = np.stack([x, rng.normal(0, 0.05, 200)], axis=1)
    segs = fit_span(pts, 0.5)
    assert len(segs) == 1 and segs[0][0] == 'L', segs


def test_line_wins_ties_on_straight():
    # jittered straight edge: nothing here should become a cubic
    x = np.linspace(0, 150, 150)
    rng = np.random.default_rng(3)
    pts = np.stack([x, rng.normal(0, 0.3, 150)], axis=1)
    pts[0, 1] = 0.0  # clean span endpoints, as corners give in practice
    pts[-1, 1] = 0.0
    segs = fit_span(pts, 1.0)
    assert segs and all(s[0] == 'L' for s in segs), segs


def test_arc_uses_cubics():
    t = np.linspace(0, np.pi / 2, 120)
    pts = np.stack([70 * np.cos(t), 70 * np.sin(t)], axis=1)
    segs = fit_span(pts, 1.0)
    assert segs and len(segs) <= 2, segs  # a clean arc needs few cubics
    assert segs[0][0] == 'C', segs


def test_fit_is_error_bounded():
    rng = np.random.default_rng(11)
    t = np.linspace(0, 2 * np.pi, 400)
    r = 60 + 8 * np.sin(3 * t)
    pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    pts += rng.normal(0, 0.2, pts.shape)
    segs = fit_span(pts, 0.8)
    assert 4 <= len(segs) <= 60, len(segs)
