
import numpy as np

def intersec(a, b):
    ymin = np.maximum(a[..., 0], b[0])
    ymax = np.minimum(a[..., 1], b[1])
    ymax = np.maximum(ymax, ymin)
    xmin = np.maximum(a[..., 2], b[2])
    xmax = np.minimum(a[..., 3], b[3])
    xmax = np.maximum(xmax, xmin)
    return np.stack((ymin, ymax, xmin, xmax), axis=-1)

def iou(a, b):
    c = intersec(a, b)
    return float(area(c)) / (area(a) + area(b) - area(c))

def area(rec):
    return (rec[..., 1] - rec[..., 0]) * (rec[..., 3] - rec[..., 2])

def box2rec(boxs):
    return boxs[..., [1,3,0,2]]

iou_box = lambda x,y: iou(box2rec(x), box2rec(y))
