import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def load_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        records: list[dict[str, str]] = json.load(file)

    if not records:
        raise ValueError("The manifest is empty.")

    return records


def create_train_val_split(
    records: list[dict[str, str]],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    train_records, val_records = train_test_split(
        records,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    return train_records, val_records


def save_split(
    train_records: list[dict[str, str]],
    val_records: list[dict[str, str]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split = {
        "train": train_records,
        "val": val_records,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(split, file, indent=2)