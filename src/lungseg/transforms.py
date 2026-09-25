from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    Spacingd,
)


def get_train_transforms(
    patch_size: tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 2,
) -> Compose:
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(
                keys=("image", "label"),
                axcodes="RAS",
            ),
            Spacingd(
                keys=("image", "label"),
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=("image",),
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=patch_size,
                pos=3,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0.0,
                allow_smaller=False,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=0,
                prob=0.5,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=1,
                prob=0.5,
            ),
            RandFlipd(
                keys=("image", "label"),
                spatial_axis=2,
                prob=0.5,
            ),
            RandRotate90d(
                keys=("image", "label"),
                prob=0.5,
                max_k=3,
            ),
            EnsureTyped(keys=("image", "label")),
        ]
    )


def get_validation_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(
                keys=("image", "label"),
                axcodes="RAS",
            ),
            Spacingd(
                keys=("image", "label"),
                pixdim=(2.0, 2.0, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=("image",),
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=("image", "label")),
        ]
    )