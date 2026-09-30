import numpy as np
import torch

from lungseg.evaluation import (
    dice_score,
    expected_calibration_error,
    predictive_entropy,
    risk_coverage_curve,
    uncertainty_error_correlation,
    voxel_variance,
)


def test_entropy_zero_for_certain_prediction():
    p = torch.zeros(1, 1, 4, 4, 4)
    e = predictive_entropy(p)
    assert e.shape == p.shape
    assert torch.all(e < 1e-3)


def test_variance_zero_for_identical_samples():
    probs = torch.rand(1, 1, 1, 8, 8, 8).repeat(4, 1, 1, 1, 1, 1)
    assert torch.allclose(voxel_variance(probs), torch.zeros_like(probs[0]))


def test_ece_perfect_calibration_is_low():
    target = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()
    assert expected_calibration_error(target.clone(), target) < 0.05


def test_ece_high_for_miscalibrated():
    target = torch.ones(1, 1, 8, 8, 8)
    assert expected_calibration_error(torch.full_like(target, 0.5), target) > 0.4


def test_ece_handles_prob_exactly_one():
    target = torch.ones(1, 1, 4, 4, 4)
    assert expected_calibration_error(torch.ones_like(target), target) < 1e-6


def test_dice_empty_vs_empty_is_one():
    z = torch.zeros(2, 1, 4, 4, 4)
    assert torch.allclose(dice_score(z, z), torch.ones(2))


def test_risk_coverage_monotone_when_confidence_is_informative():
    dice = np.array([0.9, 0.8, 0.6, 0.4, 0.2])
    rc = risk_coverage_curve(dice, confidence=dice, coverages=[0.2, 0.6, 1.0])
    assert rc["dice"] == sorted(rc["dice"], reverse=True)
    assert rc["dice"] == rc["oracle_dice"]


def test_risk_coverage_full_coverage_equals_mean_dice():
    dice = np.random.rand(7)
    rc = risk_coverage_curve(dice, confidence=np.random.rand(7), coverages=[1.0])
    assert abs(rc["dice"][0] - dice.mean()) < 1e-9


def test_entropy_error_auroc_high_when_errors_have_high_entropy():
    target = torch.zeros(1, 1, 16, 16, 16)
    p = torch.zeros_like(target)
    p[..., :4] = 0.5001  # uncertain and (barely) wrong
    r = uncertainty_error_correlation(p, target)
    assert r["auroc_entropy_error"] > 0.9