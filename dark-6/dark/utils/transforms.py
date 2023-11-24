import cv2
import numpy as np
import random
from .util import max_inscribed_rect

class Transform():
    def __init__(self):
        pass

    def __call__(self, im):
        raise Exception("Not implemented")

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
        if random.random() > self.p:
            return im

        flippedIm = cv2.flip(im, 1)
        return flippedIm
    
class FlipVertical():
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, im):
        if random.random() > self.p:
            return im

        flippedIm = cv2.flip(im, 0)
        return flippedIm
    
class Rotate():
    def __init__(self, limit, p=0.5):
        super().__init__()
        self.p = p
        self.limit = limit

    def __call__(self, im):
        if random.random() > self.p:
            return im

        imH, imW = im.shape[:2]
        angle = int(2 * self.limit * random.random() - self.limit)
        
        im = self.rotate_bound(im, angle)
        im = self.crop(im, *max_inscribed_rect(imW, imH, angle))
        im = cv2.resize(im, (imW, imH), interpolation=cv2.INTER_AREA)

        return im
    
    # https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def crop(self, img, w, h):
        xC, yC = int(img.shape[1] * .5), int(img.shape[0] * .5)

        return img[
            int(np.ceil(yC - h * .5)) : int(np.floor(yC + h * .5)),
            int(np.ceil(xC - w * .5)) : int(np.floor(xC + h * .5))
        ]


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
        if random.random() > self.p:
            return im
        
        min_s, max_s = self.sigma_limit
        sigma = random.random() * (max_s - min_s) + min_s

        im = cv2.GaussianBlur(im, self.kernel_size, sigma)
        return im

class BrightnessJitter():
    def __init__(self, brightness=(-0.2, 0.2), p=0.5):
        self.brightness_range = brightness
        self.p = p

    def __call__(self, im):
        if random.random() > self.p:
            return im
        
        min_b, max_b = self.brightness_range
        brightness_factor = random.random() * (max_b - min_b) + min_b

        im = im + brightness_factor * 255
        im = im.clip(0, 255).astype(np.uint8)
        return im         

class ContrastJitter():
    def __init__(self, contrast=(-0.2, 0.2), p=0.5):
        self.contrast_range = contrast
        self.p = p

    def __call__(self, im):
        if random.random() > self.p:
            return im
        
        min_c, max_c = self.contrast_range
        contrast_factor = random.random() * (max_c - min_c) + min_c
        
        contrast_factor += 1 
        im = im * contrast_factor
        
        im = im.clip(0, 255).astype(np.uint8)
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
