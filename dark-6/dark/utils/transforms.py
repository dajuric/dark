import cv2
import numpy as np
import random

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
    
class FlipVertical():
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, im):
        if random.random() < self.p:
            return im

        flippedIm = cv2.flip(im, 0)
        return flippedIm
    
class Rotate():
    def __init__(self, limit, p=0.5):
        super().__init__()
        self.p = p
        self.limit = limit

    def __call__(self, im):
        if random.random() < self.p:
            return im

        angle = int(2 * self.limit * random.random() - self.limit)
        h, w, _ = im.shape
        cX, cY = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotatedIm = cv2.warpAffine(im, M, (w, h))
        return rotatedIm

class Grayscale():
    def __init__(self):
        super().__init__()

    def __call__(self, im):
        resizedIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return resizedIm

class GaussianBlur():
    def __init__(self, kernel_size=(7, 7), sigma_limit=(0.01, 1.5), p=0.5):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma_limit = sigma_limit

    def __call__(self, im):
        if random.random() < self.p:
            return im
        
        min_s, max_s = self.sigma_limit
        sigma = random.random() * (max_s - min_s) + min_s

        im = cv2.GaussianBlur(im, self.kernel_size, sigma)
        return im

class JitterBrightness():
    def __init__(self, brightness=(-0.2, 0.2), p=0.5):
        self.brightness_range = brightness
        self.p = p

    def __call__(self, im):
        if random.random() < self.p:
            return im
        
        min_b, max_b = self.brightness_range
        brightness_factor = random.random() * (max_b - min_b) + min_b

        im += brightness_factor 
        return im         

class JitterContrast():
    def __init__(self, contrast=(-0.2, 0.2), p=0.5):
        self.contrast_range = contrast
        self.p = p

    def __call__(self, im):
        if random.random() < self.p:
            return im
        
        min_c, max_c = self.contrast_range
        contrast_factor = random.random() * (max_c - min_c) + min_c

        mean = im.mean()
        im = (im - mean) * contrast_factor + mean
        return im
        

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
        if im.ndim < 3: 
            im = np.expand_dims(im, -1)
            
        im = np.rollaxis(im, -1, 0) #channels first
        return im 
