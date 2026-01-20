import torch
import numpy as np
import torch.nn as nn


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def diff_x(self, input, r):
        left = input[:, :, r : 2 * r + 1]
        middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=2)
        return output

    def diff_y(self, input, r):
        left = input[:, :, :, r : 2 * r + 1]
        middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
        output = torch.cat([left, middle, right], dim=3)
        return output

    def forward(self, x):
        return self.diff_y(
            self.diff_x(x.sum(1, keepdims=True).cumsum(dim=2), self.r[1]).cumsum(dim=3),
            self.r[1],
        )


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        if isinstance(r, list):
            self.r = r
        else:
            self.r = [r, r]
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        mean_y_tmp = self.boxfilter(y)
        x, y = 0.001 * x, 0.001 * y
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        N = self.boxfilter(torch.ones_like(x))
        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N
        return (
            mean_A * x[:, c_x // 2 : c_x // 2 + 1, :, :] + mean_b
        ) / 0.001, mean_y_tmp


def fusion_perslice(
    x,
    mask,
    GFr,
    device,
):
    n, c, m, n = x.shape
    GF = GuidedFilter(r=GFr, eps=1)
    x = torch.from_numpy(x).to(device)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(device).to(torch.float)

    result, num = GF(x, mask)
    num = num == (2 * GFr[1] + 1) * (2 * GFr[1] + 1) * GFr[0]
    result[num] = 1
    result = result / result.sum(0, keepdim=True)
    minn, maxx = x.min(), x.max()
    y_seg = x[:, c // 2 : c // 2 + 1, :, :] * result
    y = torch.clip(y_seg.sum(0), minn, maxx)

    return (
        y.squeeze().cpu().data.numpy().astype(np.uint16),
        result.squeeze().cpu().data.numpy(),
    )
