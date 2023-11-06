import random
import numpy as np

class PointTransform():
    def __call__(self, pts, im_shape):
        pass
    
class Compose():
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, pts, im_shape):
        transformed_pts = pts
        
        for pt_t in self.transforms:
            transformed_pts = pt_t(transformed_pts, im_shape)

        return transformed_pts


class Resize(PointTransform):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h

    def __call__(self, pts, im_shape):
        ih, iw = im_shape[:2]
        center = np.array([iw / 2, ih / 2])
        scale = np.array([self.w / iw, self.h / ih])
        
        spts = (pts - center) / scale + center 
        return spts

class Rotate(PointTransform):
    def __init__(self, limit, p=0.5):
        super().__init__()
        self.p = p
        self.limit = limit

    def __call__(self, pts, im_shape):
        if random.random() < self.p:
            return pts

        angle = int(2 * self.limit * random.random() - self.limit)
        h, w = im_shape[:2]
        center = np.array([w / 2, h / 2])

        rads = np.radians(angle)
        R = np.array([
                    [np.cos(rads), -np.sin(rads)],
                    [np.sin(rads), +np.cos(rads)]
                    ])
        
        tpts = (pts - center) @ R.T + center
        return tpts

class Normalize(PointTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, pts, im_shape):
        h, w = im_shape[:2]
        
        return pts / np.array([w, h])
    
       