from lungseg.diffusion import DiffusionSchedule
from lungseg.models import BaselineUNet, DiffusionRefiner
from lungseg.training.joint_trainer import JointTrainer


def _mini_cfg():
    return {
        "model": {"channels": (8, 16, 32), "num_res_units": 1},
        "training": {
            "learning_rate": 1e-3, "weight_decay": 0.0,
            "boundary_weight": 0.1, "diffusion_weight": 1.0,
            "amp": False,
        },
        "joint": {"use_boundary": False, "mode": "alternating"},
        "diffusion": {"timesteps": 4, "beta_start": 1e-4, "beta_end": 0.02},
    }


def test_warmup_step_runs(sample_batch):
    cfg = _mini_cfg()
    device = sample_batch["image"].device
    baseline = BaselineUNet(channels=(8, 16, 32)).to(device)
    refiner = DiffusionRefiner(channels=(8, 16, 32)).to(device)
    sched = DiffusionSchedule(timesteps=4).to(device)

    trainer = JointTrainer(baseline, refiner, sched, cfg, device)
    loader = [sample_batch]
    loss = trainer.train_one_epoch(loader, mode="warmup")
    assert isinstance(loss, float) and loss == loss  # not NaN


def test_alternating_step_runs(sample_batch):
    cfg = _mini_cfg()
    device = sample_batch["image"].device
    baseline = BaselineUNet(channels=(8, 16, 32)).to(device)
    refiner = DiffusionRefiner(channels=(8, 16, 32)).to(device)
    sched = DiffusionSchedule(timesteps=4).to(device)
    trainer = JointTrainer(baseline, refiner, sched, cfg, device)
    loss = trainer.train_one_epoch([sample_batch], mode="alternating")
    assert loss == loss


def test_end_to_end_step_runs(sample_batch):
    cfg = _mini_cfg()
    device = sample_batch["image"].device
    baseline = BaselineUNet(channels=(8, 16, 32)).to(device)
    refiner = DiffusionRefiner(channels=(8, 16, 32)).to(device)
    sched = DiffusionSchedule(timesteps=4).to(device)
    trainer = JointTrainer(baseline, refiner, sched, cfg, device)
    loss = trainer.train_one_epoch([sample_batch], mode="end_to_end")
    assert loss == loss