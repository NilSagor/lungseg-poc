import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from lungseg.training.lightning_module import HistoryRecorder, LitSegmentation, PlainWeightsExport


def _cfg(name="unet3d"):
    return {
        "seed": 0,
        "data": {"patch_size": [32, 32, 32]},
        "model": {"name": name, "channels": [8, 16, 32], "num_res_units": 1},
        "training": {"learning_rate": 1e-3, "weight_decay": 0.0, "boundary_weight": 0.1},
        "evaluation": {"threshold": 0.5, "sw_batch_size": 1, "overlap": 0.25},
    }


def _batches(n):
    items = [{"image": torch.randn(1, 1, 32, 32, 32),
              "label": (torch.rand(1, 1, 32, 32, 32) > 0.9).float()} for _ in range(n)]
    return DataLoader(items, batch_size=None)  # items are already batches


def _trainer(callbacks, epochs):
    return L.Trainer(max_epochs=epochs, accelerator="cpu", logger=False,
                     enable_checkpointing=False, enable_progress_bar=False,
                     enable_model_summary=False, num_sanity_val_steps=0, callbacks=callbacks)


def test_fit_records_history_and_exports_plain_weights(tmp_path):
    hist = HistoryRecorder()
    trainer = _trainer([hist, PlainWeightsExport(tmp_path)], epochs=2)
    module = LitSegmentation(_cfg())
    trainer.fit(module, _batches(2), _batches(1))

    assert [r["epoch"] for r in hist.rows] == [1, 2]
    assert all(r["train_loss"] is not None and "dice" in r for r in hist.rows)
    # best.pt / last.pt are plain state_dicts that load straight into the model,
    # which is what evaluate.py and train_diffusion.py do.
    fresh = LitSegmentation(_cfg()).model
    fresh.load_state_dict(torch.load(tmp_path / "best.pt"))
    fresh.load_state_dict(torch.load(tmp_path / "last.pt"))


def test_deep_supervision_model_trains_under_lightning(tmp_path):
    hist = HistoryRecorder()
    trainer = _trainer([hist, PlainWeightsExport(tmp_path)], epochs=1)
    trainer.fit(LitSegmentation(_cfg("nnunet_style")), _batches(2), _batches(1))
    assert hist.rows and hist.rows[0]["train_loss"] > 0


def test_early_stopping_callback_stops_training(tmp_path):
    stopper = EarlyStopping(monitor="val/dice", mode="max", patience=1, min_delta=1.0)
    hist = HistoryRecorder()
    trainer = _trainer([hist, stopper], epochs=6)
    trainer.fit(LitSegmentation(_cfg()), _batches(2), _batches(1))
    assert stopper.stopped_epoch > 0 and len(hist.rows) < 6


def test_resume_keeps_prior_history_and_best(tmp_path):
    prior = [{"epoch": 1, "train_loss": 1.0, "dice": 0.99}]
    hist = HistoryRecorder(prior)
    export = PlainWeightsExport(tmp_path, best=0.99)
    trainer = _trainer([hist, export], epochs=1)
    trainer.fit(LitSegmentation(_cfg()), _batches(1), _batches(1))
    assert hist.rows[0] == prior[0] and len(hist.rows) == 2
    assert not (tmp_path / "best.pt").exists()  # a worse epoch must not overwrite the best


def test_diffusion_trains_refiner_only_and_keeps_baseline_frozen(tmp_path):
    from lungseg.models import build_from_config
    from lungseg.training.lightning_module import LitDiffusion

    cfg = _cfg()
    cfg["training"]["diffusion_weight"] = 1.0
    cfg["diffusion"] = {"timesteps": 4, "beta_start": 1e-4, "beta_end": 0.02,
                        "inference_steps": 2}
    baseline = build_from_config(cfg)
    before = {k: v.clone() for k, v in baseline.state_dict().items()}
    module = LitDiffusion(cfg, baseline)
    refiner_before = {k: v.clone() for k, v in module.refiner.state_dict().items()}

    hist = HistoryRecorder()
    trainer = _trainer([hist, PlainWeightsExport(tmp_path, attr="refiner", save_best=False)], 1)
    trainer.fit(module, _batches(2), _batches(1))

    assert all(torch.equal(before[k], v) for k, v in baseline.state_dict().items())
    assert any(not torch.equal(refiner_before[k], v) for k, v in module.refiner.state_dict().items())
    assert not module.baseline.training            # stays in eval mode after fit
    assert (tmp_path / "last.pt").exists() and not (tmp_path / "best.pt").exists()
    assert hist.rows[0]["train_loss"] is not None and "dice" in hist.rows[0]