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

clean:
	rm -rf outputs checkpoints .pytest_cache .ruff_cache