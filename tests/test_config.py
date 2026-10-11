import glob

from lungseg.config import load_config


def test_base_chaining_and_deep_merge(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "seed: 1\nmodel:\n  channels: [8, 16]\n  num_res_units: 1\n"
        "evaluation:\n  overlap: 0.25\n  sw_batch_size: 2\n")
    (tmp_path / "child.yaml").write_text(
        f"base: {tmp_path / 'base.yaml'}\nmodel:\n  name: resunet3d\n"
        "evaluation:\n  sw_batch_size: 1\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["seed"] == 1
    assert cfg["model"] == {"channels": [8, 16], "num_res_units": 1, "name": "resunet3d"}
    assert cfg["evaluation"] == {"overlap": 0.25, "sw_batch_size": 1}
    assert "base" not in cfg


def test_every_experiment_config_is_a_fair_comparison():
    ref = load_config("configs/poc.yaml")
    files = sorted(glob.glob("configs/experiments/*.yaml"))
    assert files, "no experiment configs found"
    for f in files:
        cfg = load_config(f)
        assert cfg["training"] == ref["training"], f
        assert cfg["data"] == ref["data"], f
        assert cfg["model"]["channels"] == ref["model"]["channels"], f
        for k in ("threshold", "overlap"):
            assert cfg["evaluation"][k] == ref["evaluation"][k], (f, k)