def test_image_and_label_have_same_spatial_shape(sample_batch):
    assert sample_batch["image"].shape[2:] == sample_batch["label"].shape[2:]


def test_label_is_binary(sample_batch):
    values = sample_batch["label"].unique()
    assert set(values.tolist()).issubset({0.0, 1.0})