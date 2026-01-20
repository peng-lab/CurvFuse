from .to_polar_space import cartesian_to_polar
import numpy as np
from skimage.morphology import remove_small_objects
import cv2


def fillHole(segMask):
    z, h, w = segMask.shape
    h += 2
    w += 2
    result = np.zeros(segMask.shape, dtype=bool)
    for i in range(z):
        _mask = np.pad(segMask[i], ((1, 1), (1, 1)))
        im_floodfill = 255 * (_mask.astype(np.uint8)).copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(im_floodfill, mask, seedPoint=(0, 0), newVal=255)
        result[i, :, :] = (segMask[i] + (~im_floodfill)[1:-1, 1:-1]).astype(bool)
    return result


def estimate_boundary(
    std_center,
    std_edge,
    seg,
    downsample_factor=3,
):
    xc = std_center.shape[-1] // 2
    yc = std_center.shape[-2] // 2
    meas = []

    offsets = [0, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50]

    for dx in offsets:
        for dy in offsets:
            cx = xc + dx
            cy = yc + dy
            seg_polar = cartesian_to_polar(
                seg[..., ::downsample_factor, ::downsample_factor],
                T=360,
                center=(cx, cy),
            )
            seg_polar = seg_polar.cpu().data.numpy()
            f_polar_center = cartesian_to_polar(
                std_center,
                T=360,
                center=(cx, cy),
            ).cpu().data.numpy() * (seg_polar > 0.9)
            f_polar_middle = cartesian_to_polar(
                std_edge,
                T=360,
                center=(cx, cy),
            ).cpu().data.numpy() * (seg_polar > 0.9)

            f_polar_center = f_polar_center.max(0)
            f_polar_middle = f_polar_middle.max(0)

            tmp = fillHole(f_polar_center > f_polar_middle)[0]
            tmp = remove_small_objects(tmp, 64)

            tmp_t = tmp.sum() * 0.9
            best_i = ((np.cumsum(tmp, 1).sum(0) > tmp_t) == 0).sum()
            mask_circle = np.arange(f_polar_middle.shape[-1])[None, :] < best_i
            meas.append(
                [
                    ((f_polar_center - f_polar_middle) * mask_circle).sum()
                    / ((seg_polar > 0.9) * mask_circle).sum(),
                    cx,
                    cy,
                    best_i,
                ]
            )
    meas = np.stack(meas)
    return meas
