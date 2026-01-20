import numpy as np
from .to_polar_space import cartesian_to_polar
from skimage.morphology import disk
from scipy.ndimage import binary_dilation, binary_opening


def mask_circle_func(best_i, cx, cy, img_center, seg_2, scale):
    T = 360  # radius sampling
    half = 9 // 2  # window size
    margin_pix = 3
    max_fg_frac = 0.03
    smooth_k = nine = 9
    prefer_outward = True

    seg_ds = seg_2[..., ::scale, ::scale]
    fg = (seg_ds > 0) * (seg_ds < 2)
    if margin_pix > 0:
        fg = binary_dilation(fg, structure=disk(margin_pix)[None]).astype(np.float32)[
            :, None
        ]

    # to polar space
    fg_polar_vol = cartesian_to_polar(fg, T=T, center=(cx, cy)).cpu().numpy()
    mask_circle_vol = np.zeros(
        img_center[..., ::scale, ::scale].shape, dtype=np.float32
    )

    for ind, fg_polar in enumerate(fg_polar_vol):
        fg_polar = fg_polar[0] > 0  # [T,R]
        T_eff, R = fg_polar.shape

        r0 = int(best_i)
        r0 = np.clip(r0, half, R - half - 1)
        r_adj = np.full((T_eff,), r0, dtype=int)

        def ok_at(strip, r):
            if r - half < 0 or r + half >= R:
                return False
            win = strip[r - half : r + half + 1]
            return (win.sum() / win.size) <= max_fg_frac

        # look for clean region for each angle
        for t in range(T_eff):
            strip = fg_polar[t].astype(np.uint8)
            if not ok_at(strip, r0):
                found = False
                max_d = max(R - r0 - 1, r0)
                for d in range(1, max_d + 1):
                    cand = []
                    L = r0 - d
                    Rr = r0 + d
                    if L >= 0 and ok_at(strip, L):
                        cand.append(("L", L))
                    if Rr < R and ok_at(strip, Rr):
                        cand.append(("R", Rr))
                    if cand:
                        if prefer_outward:
                            cand.sort(key=lambda x: (abs(x[1] - r0), x[0] != "R"))
                        else:
                            cand.sort(key=lambda x: (abs(x[1] - r0), x[0] != "L"))
                        r_adj[t] = cand[0][1]
                        found = True
                        break
                if not found:
                    r_adj[t] = np.clip(r0, half, R - half - 1)

        # smoothing
        if smooth_k % 2 == 1 and smooth_k > 1:
            pad = smooth_k // 2
            r_pad = np.pad(r_adj, (pad, pad), mode="wrap")
            ker = np.ones(smooth_k, dtype=np.float32) / smooth_k
            r_adj = np.convolve(r_pad, ker, mode="valid").astype(int)

        # back to image space
        H, W = img_center[:, :, ::scale, ::scale].shape[-2:]
        yy, xx = np.mgrid[0:H, 0:W]
        ang = np.arctan2(yy - cy, xx - cx)  # [-pi, pi]
        t_idx = np.floor((ang + np.pi) / (2 * np.pi) * T_eff).astype(int)
        t_idx = np.clip(t_idx, 0, T_eff - 1)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask_circle = (rr <= r_adj[t_idx]).astype(np.uint8)

        # smoothing
        mask_circle = binary_opening(mask_circle, structure=disk(1)).astype(np.uint8)

        mask_circle_vol[ind] = mask_circle

    return mask_circle_vol
