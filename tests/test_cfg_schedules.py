"""CPU checks for guidance constraints and time-dependent scale normalization."""

import pytest
import torch

from cfg_norm_clamped_eval.cfg_schedules import TVCFG, norm_clamped, sd35_norm_clamped


@pytest.mark.parametrize("sigma", [0.0, 0.2, 0.9, 1.0])
@pytest.mark.parametrize("gamma", [1.0, 1.1, 1.5])
def test_guidance_constraints(sigma, gamma):
    generator = torch.Generator().manual_seed(42)
    xt, cond, uncond = [torch.randn(8, 4, 8, 8, generator=generator) for _ in range(3)]
    gap = cond - uncond
    for schedule in (norm_clamped, sd35_norm_clamped):
        weight = schedule(sigma, gap=gap, v_cond=cond, xt=xt, cfg_scale=5, gamma=gamma)
        assert torch.isfinite(weight).all()
        assert ((weight >= 1) & (weight <= 5)).all()
        alpha = (weight - 1).view(-1, 1, 1, 1)
        if schedule is norm_clamped:
            conditional = xt + (1 - sigma) * cond
            guided = conditional + alpha * (1 - sigma) * gap
            bound = gamma * conditional.flatten(1).norm(dim=1)
            assert (guided.flatten(1).norm(dim=1) <= bound + 1e-4).all()
        else:
            conditional = xt - sigma * cond
            change = -alpha * sigma * gap
            bound = (gamma - 1) * conditional.flatten(1).norm(dim=1)
            assert (change.flatten(1).norm(dim=1) <= bound + 1e-4).all()


def test_zero_gap_preserves_existing_conventions():
    xt = torch.ones(2, 4, 2, 2)
    kwargs = dict(gap=torch.zeros_like(xt), v_cond=xt, xt=xt, cfg_scale=5)
    torch.testing.assert_close(norm_clamped(0.5, **kwargs), torch.ones(2))
    torch.testing.assert_close(sd35_norm_clamped(0.5, **kwargs), torch.full((2,), 5.0))


@pytest.mark.parametrize("times", [[0, 0.1, 0.4, 1], [1, 0.7, 0.1, 0], [0, 1]])
def test_tv_cfg_has_requested_time_weighted_mean(times):
    schedule = TVCFG(times, 4)
    mean = sum(
        schedule((a + b) / 2) * width
        for a, b, width in zip(times, times[1:], schedule.normalized_widths)
    )
    assert mean == pytest.approx(4)
    assert schedule(times[-1]) == schedule.scales[-1]
    assert schedule(times[0]) == schedule.scales[0]


@pytest.mark.parametrize("times", [[0], [0, 0], [0, 1, 0.5], [0, float("nan")]])
def test_tv_cfg_rejects_invalid_grids(times):
    with pytest.raises(ValueError):
        TVCFG(times, 4)


@pytest.mark.parametrize("scale", [0.5, float("inf"), float("nan")])
def test_tv_cfg_rejects_invalid_scales(scale):
    with pytest.raises(ValueError):
        TVCFG([0, 1], scale)
