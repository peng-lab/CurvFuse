import numpy as np
from .feature_extraction import NSCTdec
from .boundary_estimation import estimate_boundary
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from .boundary_refinement import mask_circle_func
import tqdm
import torch
import torch.nn.functional as F
from .guided_filtering_fusion import fusion_perslice


def curvfuse(
    input_center,
    input_edge,
    downsample_factor=3,
    decompose_levels=[5, 5, 5, 5, 5, 5],
    std_size=15,
    gf_size=49,
    device="cuda",
):

    seg = np.zeros(input_center.shape, dtype=np.float32)
    seg += (input_center > threshold_otsu(input_center)).astype(np.float32)
    seg += (input_edge > threshold_otsu(input_edge)).astype(np.float32)
    seg_2 = remove_small_objects(
        (input_center > threshold_otsu(input_center)), 64
    ).astype(np.float32) + remove_small_objects(
        (input_edge > threshold_otsu(input_edge)), 64
    ).astype(
        np.float32
    )

    nsct_dec = NSCTdec(
        levels=decompose_levels,
        device=device,
        r=std_size,
    )

    std_center = np.zeros(
        (
            input_center.shape[0],
            input_center.shape[-2] // downsample_factor,
            input_center.shape[-1] // downsample_factor,
        ),
        dtype=np.float32,
    )
    std_edge = np.zeros(
        (
            input_edge.shape[0],
            input_edge.shape[-2] // downsample_factor,
            input_edge.shape[-1] // downsample_factor,
        ),
        dtype=np.float32,
    )

    for i in range(input_center.shape[0]):
        _, _, std_center[i] = nsct_dec(
            input_center[i : i + 1, None],
            stride=downsample_factor,
            _forFeatures=True,
        )
        _, _, std_edge[i] = nsct_dec(
            input_edge[i : i + 1, None],
            stride=downsample_factor,
            _forFeatures=True,
        )

    meas = estimate_boundary(
        std_center,
        std_edge,
        seg,
        downsample_factor=downsample_factor,
    )

    meas_max = meas[np.argmax(meas[:, 0]), 1:]
    cx = meas_max[0]
    cy = meas_max[1]
    r_out = meas_max[-1]
    best_i = r_out

    mask_circle = mask_circle_func(
        best_i,
        cx,
        cy,
        input_center,
        seg_2,
        downsample_factor,
    )

    result = np.zeros_like(input_center)

    for i in tqdm.tqdm(range(input_center.shape[0])):
        result[i], _ = fusion_perslice(
            np.stack(
                (
                    input_center[i],
                    input_edge[i],
                )
            )[:, None],
            F.interpolate(
                torch.from_numpy(
                    np.stack((mask_circle[i], 1 - mask_circle[i, 0]))[:, None].astype(
                        np.float32
                    )
                ),
                input_center.shape[-2:],
            )
            .cpu()
            .data.numpy(),
            [gf_size // 2, gf_size // 2],
            device,
        )
    return result
