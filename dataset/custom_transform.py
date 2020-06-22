# from __future__ import division
import random

import numpy as np
from PIL import Image, ImageEnhance

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''

'''I refered from DPSNET custom transform'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


# normalize
class normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        # normalize for Flickr dataset
        # magic method of numpy function
        mean = np.array(self.mean).reshape((3, 1, 1))
        std = np.array(self.std).reshape((3, 1, 1))
        images = (images - mean) / std
        return images


class ScaleDown(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __init__(self, out_size):
        self.out_h, self.out_w = out_size

    def __call__(self, images):
        scaled_images = list()
        for image in images:
            in_w, in_h = image.size
            x_scaling = self.out_w / in_w
            y_scaling = self.out_h / in_h
            scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

            scaled_image = image.resize((scaled_w, scaled_h))
            scaled_images.append(scaled_image)
        return scaled_images


class RandomRotation(object):
    '''Randomly Rotation degrees chosen between degrees variables'''

    # rerefence : https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomRotation

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        if isinstance(degrees, (float, int)):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, images):
        angle = np.random.uniform(self.degrees)
        rotated = list()
        for im in images:
            rotated.append(im.rotate(angle, self.resample, self.expand, self.center, fillcolor=self.fill))
        return rotated

# from DPSNET
class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __init__(self, scaleSize):
        self.out_h, self.out_w = scaleSize

    def __call__(self, images):
        cropped_images = list()
        for im in images:
            in_w, in_h = im.size
            # sample using uniform distribution
            x_scaling = np.random.uniform(self.out_w / in_w, 1)
            y_scaling = np.random.uniform(self.out_h / in_h, 1)
            scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

            scaled_image = im.resize((scaled_w, scaled_h))

            offset_y = np.random.randint(scaled_h - self.out_h + 1)
            offset_x = np.random.randint(scaled_w - self.out_w + 1)

            cropped_image = scaled_image[offset_y:offset_y + self.out_h, offset_x:offset_x + self.out_w, :]
            cropped_images.append(cropped_image)

        return cropped_images


# reference https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=colorjitter#torchvision.transforms.ColorJitter
class ColorJitter(object):
    ''' Randomly change the brightness, contrast and saturation of an image'''
    ''' the value must be between 0 ~ 0.5'''

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = [max(0, 1 - brightness), max(0, 1 + brightness)]
        self.contrast = [max(0, 1 - contrast), max(0, 1 + contrast)]
        self.saturation = [max(0, 1 - saturation), max(0, 1 + saturation)]
        self.hue = [-hue, hue]

    def random_select(self, brightness, contrast, saturation, hue):
        brightness = np.random.uniform(brightness[0], brightness[1])
        contrast = np.random.uniform(contrast[0], contrast[1])
        saturation = np.random.uniform(saturation[0], saturation[1])
        hue = np.random.uniform(hue[0], hue[1])
        return brightness, contrast, saturation, hue

    # default use PIL classes
    def __call__(self, images):
        brightness, contrast, saturation, hue = self.random_select(self.brightness,
                                                                   self.contrast,
                                                                   self.saturation,
                                                                   self.hue)
        im_list = []
        for im in images:
            # adjust brightness
            im = ImageEnhance.Brightness(im).enhance(brightness)
            # adjust contrast
            im = ImageEnhance.Contrast(im).enhance(contrast)
            # adjust color
            im = ImageEnhance.Color(im).enhance(saturation)
            # adjust hue
            h, s, v = im.convert('HSV').split()
            np_h = np.array(h, dtype=np.uint8)
            with np.errstate(over='ignore'):
                np_h += np.uint8(hue * 255)
            h = Image.fromarray(np_h, 'L')
            im = Image.merge('HSV', (h, s, v)).convert('RGB')
            im_list.append(im)
        return im_list


# https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomHorizontalFlip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        flipped = list()
        for im in images:
            if random.random() < self.p:
                flipped.append(im.transpose(Image.FLIP_LEFT_RIGHT))
            else:
                flipped.append(im)
        return flipped


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        flipped = list()
        for im in images:
            if random.random() < self.p:
                flipped.append(im.transpose(Image.FLIP_TOP_BOTTOM))
            else:
                flipped.append(im)
        return flipped


# from DPSNET
class CropCenter(object):
    """crop the image center before"""

    def __call__(self, images):

        # crop size for x and y
        cropx = 320
        cropy = 320

        # where to start crop
        y, x, _ = images[0].shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)

        # change image scales
        crop_images = [im[starty:starty + cropy, startx:startx + cropx, :] for im in images]
        crop_depth = depth[starty:starty + cropy, startx:startx + cropx]

        x_scale = cropx / x
        y_scale = cropy / y
        # change intrinsics's center point
        output_intrinsics[0] *= x_scale
        output_intrinsics[1] *= y_scale

        return crop_images


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        # change list to numpy array
        for i, im in enumerate(images):
            images[i] = np.array(im)
        images = np.stack(images, axis=0)
        # normalize with 255
        images = images / 255.0
        transposed = images.transpose((0, 3, 1, 2))
        return transposed


class Normalize(object):
    def __call__(self, images):
        # normalize for mnist dataset
        images = images / 255.0
        return images
