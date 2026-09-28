
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose


def create_train_loader(
    records: list[dict[str, str]],
    transform: Compose,
    batch_size: int = 1,
    num_workers: int = 2,
    cache_rate: float = 0.5,
) -> DataLoader:
    dataset = CacheDataset(
        data=records,
        transform=transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
        copy_cache=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def create_validation_loader(
    records: list[dict[str, str]],
    transform: Compose,
    num_workers: int = 2,
    cache_rate: float = 1.0,
) -> DataLoader:
    dataset = CacheDataset(
        data=records,
        transform=transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
        copy_cache=False,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )