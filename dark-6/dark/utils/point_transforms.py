import random
import numpy as np
from .util import max_inscribed_rect

class PointTransform():
    def __call__(self, pts, im_shape):
        pass
    
class Compose():
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, pts, im_shape):
        t_pts = pts
        t_shape = im_shape[:2]
        
        for pt_t in self.transforms:
            t_pts, t_shape = pt_t(t_pts, t_shape)

        return t_pts, t_shape


class Resize(PointTransform):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h

    def __call__(self, pts, im_shape):
        ih, iw = im_shape[:2]
        scale = np.array([self.w / iw, self.h / ih])
        
        spts = pts * scale
        return spts, (self.h, self.w)

class Rotate(PointTransform):
    def __init__(self, limit, p=0.5):
        super().__init__()
        self.p = p
        self.limit = limit

    def __call__(self, pts, im_shape):
        if random.random() > self.p:
            return pts, im_shape

        angle = int(2 * self.limit * random.random() - self.limit)
        h, w = im_shape[:2]
        center = np.array([w / 2, h / 2])

        rads = np.radians(angle)
        R = np.array([
                    [np.cos(rads), -np.sin(rads)],
                    [np.sin(rads), +np.cos(rads)]
                    ])
        
        # transform and translate using new centre
        nW, nH = max_inscribed_rect(w, h, angle)  
        tpts = (pts - center) @ R.T + np.array((nW/2, nH/2)) 

        #rescale to original image size
        s = np.array((w / nW, h / nH))
        tpts *= s

        return tpts, im_shape
        
class Normalize(PointTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, pts, im_shape):
        h, w = im_shape[:2]
        
        n_pts = pts / np.array([w, h])
        return n_pts, im_shape
    
       