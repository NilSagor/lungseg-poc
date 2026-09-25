import json
from pathlib import Path



def build_task06_manifest(data_root: str | Path) -> list[dict[str, str]]:
    data_root = Path(data_root)
    image_dir = data_root / "imagesTr"
    label_dir = data_root / "labelsTr"

    image_files = sorted(image_dir.glob("*.nii.gz"))
    records: list[dict[str, str]] = []

    for image_path in image_files:
        label_path = label_dir / image_path.name

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for {image_path.name}: {label_path}"
            )

        records.append(
            {
                "image": str(image_path),
                "label": str(label_path),
            }
        )

    if not records:
        raise RuntimeError(f"No training images found in {image_dir}")

    return records


def save_manifest(
    records: list[dict[str, str]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

