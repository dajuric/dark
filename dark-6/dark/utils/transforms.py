import cv2
import numpy as np
import random
import dark.tensor as xp

class Transform():
    def __init__(self):
        pass

    def __call__(self, im):
        raise NotImplemented()

class Compose():
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, im):
        transformedIm = im
        for imT in self.transforms:
            transformedIm = imT(transformedIm)

        return transformedIm


class Resize():
    def __init__(self, imW, imH):
        super().__init__()
        self.imW = imW
        self.imH = imH

    def __call__(self, im):
        resizedIm = cv2.resize(im, (self.imW, self.imH), interpolation=cv2.INTER_LINEAR)
        return resizedIm

class FlipHorizontal():
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, im):
        if random.random() < self.p:
            return im

        flippedIm = cv2.flip(im, 1)
        return flippedIm

class Grayscale():
    def __init__(self):
        super().__init__()

    def __call__(self, im):
        resizedIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return resizedIm

class Normalize():
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0):
         super().__init__()
         self.mean = mean
         self.std = std
         self.max_pixel_value = max_pixel_value

    def __call__(self, im):
        _mean = np.array(self.mean) * self.max_pixel_value
        _std  = np.array(self.std)  * self.max_pixel_value

        resultIm = (im - _mean) / _std
        return resultIm

class ToTensorV2():
    def __init__(self):
        super().__init__()

    def __call__(self, im):
        im = np.rollaxis(im, -1, 0) #channels first
        if im.ndim < 3: im = np.expand_dims(im, 0)
        
        im = xp.asarray(im) #convert to CUDA array is available
        return im 
