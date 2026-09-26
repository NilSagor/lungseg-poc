"""Discover MSD Task06 Lung cases and write a patient-level split."""
import argparse
from lungseg.data import discover_cases, make_patient_split, save_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="data/splits/train_val.json")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cases = discover_cases(a.root)
    print(f"Discovered {len(cases)} labeled cases under {a.root}")
    split = make_patient_split(cases, val_fraction=a.val_fraction, seed=a.seed)
    save_split(split, a.out)
    print(f"Wrote split: train={len(split['train'])}, val={len(split['val'])} -> {a.out}")


if __name__ == "__main__":
    main()