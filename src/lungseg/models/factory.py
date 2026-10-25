from .baselines import (
    BaselineUNet,
    NNUNetStyle,
    ResUNet3D,
    SegResNetSeg,
    SwinUNETRSeg,
    UNETRSeg,
)

SEGMENTATION_MODELS = (
    "unet3d",
    "resunet3d",
    "nnunet_style",
    "swinunetr",
    "unetr",
    "segresnet",
)

# _TRANSFORMER_VARIANTS = ("swinunetr", "unetr")


def build_segmentation_model(
    name: str,
    channels=(16, 32, 64, 128),
    num_res_units: int = 1,
    patch_size=(96, 96, 96),
    feature_size: int | None = None,
    variant: str | None = None,
):
    name = name.lower()
    channels = tuple(channels)

    if name == "unet3d":
        return BaselineUNet(channels=channels, num_res_units=num_res_units)

    if name == "resunet3d":
        return ResUNet3D(channels=channels)

    if name == "nnunet_style":
        # Phase 12 will fill the implementation; the call site is fixed now.
        return NNUNetStyle(channels=channels, patch_size=tuple(patch_size))
    

    # if name == "transformer":
    #     variant = (variant or "swinunetr").lower()
    #     if variant == "swinunetr":
    #         return SwinUNETRSeg(feature_size=feature_size or 24)
    #     if variant == "unetr":
    #         return UNETRSeg(img_size=tuple(patch_size), feature_size=feature_size or 16)
    #     raise ValueError(f"Unknown transformer variant {variant!r}; choose from {_TRANSFORMER_VARIANTS}")

    if name == "swinunetr":
        return SwinUNETRSeg(feature_size=feature_size or 24)

    if name == "unetr":
        return UNETRSeg(
            img_size=tuple(patch_size),
            feature_size=feature_size or 16,
        )
    
    
    if name == "segresnet":
        return SegResNetSeg(init_filters=channels[0])

    raise ValueError(f"Unknown segmentation model {name!r}; choose from {SEGMENTATION_MODELS}")


def build_from_config(cfg: dict):
    """Build the segmentation network described by ``cfg['model']``."""
    m = cfg["model"]
    return build_segmentation_model(
        name=m.get("name", "unet3d"),
        channels=m["channels"],
        num_res_units=m.get("num_res_units", 1),
        patch_size=cfg["data"]["patch_size"],
        feature_size=m.get("feature_size"),
        variant=m.get("variant"),
    )