# Lung Tumor Segmentation with Conditional Diffusion Refinement (PoC)

Proof of concept: *Can a conditional diffusion-assisted 3D segmentation framework
improve lung-tumor segmentation, especially for small, irregular, low-contrast tumors?*

## Pipeline
NIfTI -> MONAI -> tumor-aware patch sampler -> 3D U-Net -> (optional) diffusion refiner -> metrics

## Experiments
| ID | Description |
|----|-------------|
| E0 | 3D U-Net (Dice + BCE) |
| E1 | 3D U-Net + boundary loss |
| E2 | 3D U-Net + diffusion refiner |
| E3 | 3D U-Net + diffusion refiner + boundary loss |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
python scripts/prepare_data.py --root data/raw/Task06_Lung
make overfit
make baseline
make diffusion
make eval
```

