import yaml
import os

import albumentations as A


# Class for normal augmentations
class Augmentation:
    def __init__(self, config):
        if isinstance(config, dict):
            config = config
        elif os.path.isfile(config):
            with open(config) as f:
                config = yaml.safe_load(f)
        else:
            raise TypeError
        self.config = config

    # load corresponding Albumentation transform function with config
    def define_aug(self):
        # Recursively compose transform
        self.augs = {}
        if self.config["albumentation"]["blur"]["prob"] > 0:
            self.blur = A.Blur(
                blur_limit=self.config["albumentation"]["blur"]["strength"],
                p=self.config["albumentation"]["blur"]["prob"]
            )
            self.augs.update({
                "blur": self.blur
            })
        if self.config["albumentation"]["clahe"]["prob"] > 0:
            self.CLAHE = A.CLAHE(
                clip_limit=self.config["albumentation"]["clahe"]["clip_limit"],
                tile_grid_size=self.config["albumentation"]["clahe"]["tile_grid_size"],
                p=self.config["albumentation"]["clahe"]["prob"]
            )
            self.augs.update({
                "clahe": self.CLAHE
            })
        if self.config["albumentation"]["color_jitter"]["prob"] > 0:
            self.color_jitter = A.ColorJitter(
                brightness=self.config["albumentation"]["color_jitter"]["brightness"],
                contrast=self.config["albumentation"]["color_jitter"]["contrast"],
                saturation=self.config["albumentation"]["color_jitter"]["saturation"],
                hue=self.config["albumentation"]["color_jitter"]["hue"],
                p=self.config["albumentation"]["color_jitter"]["prob"]
            )
            self.augs.update({
                "color_jitter": self.color_jitter
            })
        if self.config["albumentation"]["coarse_dropout"]["prob"] > 0:
            self.cutout = A.CoarseDropout(
                max_holes=self.config["albumentation"]["coarse_dropout"]["max_holes"],
                max_height=self.config["albumentation"]["coarse_dropout"]["max_height"],
                max_width=self.config["albumentation"]["coarse_dropout"]["max_width"],
                min_holes=self.config["albumentation"]["coarse_dropout"]["min_holes"],
                min_height=self.config["albumentation"]["coarse_dropout"]["min_height"],
                min_width=self.config["albumentation"]["coarse_dropout"]["min_width"],
                fill_value=self.config["albumentation"]["coarse_dropout"]["fill_value"],
                p=self.config["albumentation"]["coarse_dropout"]["prob"]
            )
            self.augs.update({
                "cutout": self.cutout
            })
        if self.config["albumentation"]["downscale"]["prob"] > 0:
            self.downscale = A.Downscale(
                scale_min=self.config["albumentation"]["downscale"]["scale_min"],
                scale_max=self.config["albumentation"]["downscale"]["scale_max"],
                p=self.config["albumentation"]["downscale"]["prob"]
            )
            self.augs.update({
                "downscale": self.downscale
            })
        if self.config["albumentation"]["affine"]["prob"] > 0:
            self.affine = A.Affine(
                scale=self.config["albumentation"]["affine"]["scale"],
                translate_percent=self.config["albumentation"]["affine"]["translate_percent"],
                rotate=self.config["albumentation"]["affine"]["rotate"],
                shear=self.config["albumentation"]["affine"]["shear"],
                fit_output=self.config["albumentation"]["affine"]["fit_output"],
                p=self.config["albumentation"]["affine"]["prob"]
            )
            self.augs.update({
                "affine": self.affine
            })
        if self.config["albumentation"]["fliplr"]["prob"] > 0:
            self.fliplr = A.HorizontalFlip(
                p=self.config["albumentation"]["fliplr"]["prob"]
            )
            self.augs.update({
                "fliplr": self.fliplr
            })

        list_transforms = [self.augs[transform] for transform in self.augs.keys()]
        self.transforms = A.Compose(list_transforms)

    def __call__(self, image):
        if self.augs:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        return image

