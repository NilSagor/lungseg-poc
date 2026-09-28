import argparse

from lungseg.data_manifest import build_task06_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = build_task06_manifest(args.data_root)
    save_manifest(records, args.output)

    print(f"Saved {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()