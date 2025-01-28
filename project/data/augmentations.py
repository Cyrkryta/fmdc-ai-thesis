import random

import numpy as np
import volumentations.augmentations.functional as F


class Transform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, targets):
        self.reshuffle(next(iter(targets.values())))

        if random.random() < self.p:
            for k, v in targets.items():
                targets[k] = self.apply(v)

        return targets

    def apply(self, volume):
        raise NotImplementedError

    def reshuffle(self, volume):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms, p=1.0):
        self.transforms = [Float()] + transforms + [Contiguous()]
        self.p = p

    def __call__(self, volumes):
        data = volumes

        for tr in self.transforms:
            data = tr(data)

        return data


class Float(Transform):
    def apply(self, image):
        return image.astype(np.float32)

    def reshuffle(self, volume):
        pass


class Contiguous(Transform):
    def apply(self, image):
        return np.ascontiguousarray(image)

    def reshuffle(self, volume):
        pass


class Rotate(Transform):
    def __init__(self, x_limit=(-15,15), y_limit=(-15,15), z_limit=(-15,15), interpolation=1, border_mode='constant', value=0, mask_value=0, always_apply=False, p=0.5):
        super().__init__(p)
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.z_limit = z_limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.params = {}

    def apply(self, volume):
        return F.rotate3d(volume, self.params['x'], self.params['y'], self.params['z'], interpolation=self.interpolation, border_mode=self.border_mode, value=self.value)

    def reshuffle(self, volume):
        self.params = {
            'x': random.uniform(self.x_limit[0], self.x_limit[1]),
            'y': random.uniform(self.y_limit[0], self.y_limit[1]),
            'z': random.uniform(self.z_limit[0], self.z_limit[1]),
        }


class ElasticTransform(Transform):
    def __init__(self, deformation_limits=(0, 0.25), interpolation=1, border_mode='constant', value=0, mask_value=0, p=0.5):
        super().__init__(p)
        self.deformation_limits = deformation_limits
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.params = {}

    def apply(self, volume):
        return F.elastic_transform(volume, self.params['sigmas'], self.params['alphas'], interpolation=self.interpolation, random_state=self.params['random_state'], border_mode=self.border_mode, value=self.value)

    def reshuffle(self, volume):
        random_state = random.randint(0, 10000)
        deformation = random.uniform(*self.deformation_limits)
        sigmas = [deformation * x for x in volume.shape[:3]]
        alphas = [random.uniform(x/8, x/2) for x in sigmas]
        self.params = {
            'random_state': random_state,
            'sigmas': sigmas,
            'alphas': alphas
        }


class RandomScale(Transform):
    def __init__(self, scale_limit=[0.9, 1.1], interpolation=1, p=0.5):
        super().__init__(p)
        self.scale_limit = scale_limit
        self.interpolation = interpolation
        self.params = {}

    def apply(self, volume):
        return F.rescale(volume, self.params['scale'], interpolation=self.interpolation)

    def reshuffle(self, volume):
        self.params = {'scale': random.uniform(self.scale_limit[0], self.scale_limit[1])}


class Resize(Transform):
    def __init__(self, shape, interpolation=1, resize_type=1, p=1):
        super().__init__(p)
        self.shape = shape
        self.interpolation = interpolation
        self.resize_type = resize_type

    def apply(self, volume):
        if volume.shape[0] == 1:
            shape = (1, *self.shape[:2])
        else:
            shape = self.shape

        return F.resize(volume, new_shape=shape, interpolation=self.interpolation, resize_type=self.resize_type)

    def reshuffle(self, volume):
        pass
