import albumentations as albu

AUGMENTATIONS = {
    "AFFINE": albu.Affine,
    "BLUR": albu.Blur,
    "COLORJITTER": albu.ColorJitter,
    "CUTOUT": albu.CoarseDropout,
    "DOWNSCALE": albu.Downscale,
    "EMBOSS": albu.Emboss,
    "GAUSS_NOISE": albu.GaussNoise,
    "GAUSS_BLUR": albu.GaussianBlur,
    "HORIZONTAL_FLIP": albu.HorizontalFlip,
    "IMAGE_COMPRESSION": albu.ImageCompression,
    "ONE_OF": albu.OneOf,
    "RANDOM_BRIGHTNESS_CONTRAST": albu.RandomBrightnessContrast,
    "RANDOM_CROP": albu.RandomCrop,
    "RESIZE": albu.Resize,
    "ROTATE": albu.Rotate,
    "SHARPEN": albu.Sharpen,
    "SHIFT_SCALE_ROTATE": albu.ShiftScaleRotate,
    "TO_GRAY": albu.ToGray,
    "VERTICAL_FLIP": albu.VerticalFlip,
}


def build_augment_compose(augmentations_dict: dict) -> albu.Compose:
    """Build the composed augmentations

    Parameters
    ----------
    augmentations_dict : dict
        The augmentations dict, extracted from the config

    Returns
    -------
    albu.Compose
        The composed augmentations

    Example
    -------
    augmentation_dict = {
        "SHIFT_SCALE_ROTATE": {
            "shift_limit": 0.05,
            "scale_limit": 0.1,
            "rotate_limit": 45,
            "border_mode": 0,
            "p": 0.7
        },
        "CUTOUT": {
            "num_holes_range": [1, 1],
            "hole_height_range": [32, 32],
            "hole_width_range": [32, 32],
            "fill_value": 0,
            "p": 0.8
        },
        "BBOX_PARAMS": {
            "format": "pascal_voc",
            "label_fields": ["category_id"]
        }
    }
    augment_compose = build_augment_compose(augmentation_dict)
    """

    augmentation_list = [
        AUGMENTATIONS[k](**augmentations_dict[k])
        for k in augmentations_dict.keys()
        if k not in ["ONE_OF", "BBOX_PARAMS", "KEYPOINTS_PARAMS"]
    ]

    if "ONE_OF" in augmentations_dict.keys():
        augmentation_one_of = [
            AUGMENTATIONS[k](**augmentations_dict["ONE_OF"]["AUG"][k])
            for k in augmentations_dict["ONE_OF"]["AUG"].keys()
        ]
        augmentation_list += [AUGMENTATIONS["ONE_OF"](augmentation_one_of, augmentations_dict["ONE_OF"]["p"])]

    # check for bbox params
    bbox_params = None
    if "BBOX_PARAMS" in augmentations_dict.keys():
        bbox_params = albu.BboxParams(**augmentations_dict["BBOX_PARAMS"])

    # check for keypoint params
    keypoints_params = None
    if "KEYPOINTS_PARAMS" in augmentations_dict.keys():
        keypoints_params = albu.KeypointParams(**augmentations_dict["KEYPOINTS_PARAMS"])

    transforms = albu.Compose(
        augmentation_list,
        bbox_params=bbox_params,
        keypoint_params=keypoints_params,
    )

    return transforms
