.PHONY: install lint test overfit baseline diffusion eval clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src scripts tests

test:
	pytest -q

overfit:
	python scripts/train_baseline.py --config configs/poc.yaml --debug --max_epochs 50

baseline:
	python scripts/train_baseline.py --config configs/poc.yaml

diffusion:
	python scripts/train_diffusion.py --config configs/poc.yaml \
		--baseline-ckpt outputs/baseline/best.pt

eval:
	python scripts/evaluate.py --config configs/poc.yaml \
		--baseline-ckpt outputs/baseline/best.pt \
		--out outputs/eval_baseline.json

experiments:
	bash scripts/run_experiments.sh

benchmark-sampling:
	python scripts/benchmark_sampling.py --config configs/poc.yaml

joint-warmup:
	python scripts/train_joint.py --config configs/poc.yaml --mode warmup \
		--baseline-ckpt outputs/e0_unet/best.pt --out outputs/joint_warmup

joint-alt:
	python scripts/train_joint.py --config configs/poc.yaml --mode alternating \
		--baseline-ckpt outputs/e0_unet/best.pt --out outputs/joint_alt

joint-e2e:
	python scripts/train_joint.py --config configs/poc.yaml --mode end_to_end \
		--baseline-ckpt outputs/e0_unet/best.pt --out outputs/joint_e2e

multiseed:
	bash scripts/run_multiseed.sh

# uncertainty:
# 	python scripts/evaluate_uncertainty.py --config configs/poc.yaml \
# 		--baseline-ckpt outputs/e0_unet_seed0/best.pt \
# 		--refiner-ckpt outputs/e2_diffusion_seed0/last.pt \
# 		--out outputs/e2_diffusion_seed0_uncertainty

uncertainty:
	python scripts/evaluate_uncertainty.py --config configs/poc.yaml \
		--baseline-ckpt outputs/e0_unet_seed$(SEED0)/best.pt \
		--refiner-ckpt outputs/e2_diffusion_seed$(SEED0)/last.pt \
		--out outputs/e2_diffusion_seed$(SEED0)_uncertainty


aggregate-multiseed:
	python scripts/aggregate_multiseed.py


clean:
	rm -rf outputs checkpoints .pytest_cache .ruff_cache