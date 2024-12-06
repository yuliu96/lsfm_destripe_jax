import jax.numpy as jnp


class BoxFilter_3d:
    def __init__(self, r):
        self.r = r

    def diff_x(self, input, r):
        left = input[:, :, r : 2 * r + 1]
        middle = input[:, :, 2 * r + 1 :] - input[:, :, : -2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1 : -r - 1]
        output = jnp.concatenate([left, middle, right], 2)
        return output

    def diff_y(self, input, r):
        left = input[:, :, :, r : 2 * r + 1]
        middle = input[:, :, :, 2 * r + 1 :] - input[:, :, :, : -2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1 : -r - 1]
        output = jnp.concatenate([left, middle, right], 3)
        return output

    def __call__(self, x):
        return self.diff_y(
            self.diff_x(x.sum(1, keepdims=True).cumsum(2), self.r[1]).cumsum(3),
            self.r[1],
        )


class GuidedFilter:
    def __init__(self, r, eps=1e-8):
        if isinstance(r, list):
            self.r = r
        else:
            self.r = [r, r]
        self.eps = eps
        self.boxfilter = BoxFilter_3d(r)

    def __call__(self, x, y):
        mean_y_tmp = self.boxfilter(y)
        x, y = 0.001 * x, 0.001 * y
        n_x, c_x, h_x, w_x = x.shape
        N = self.boxfilter(jnp.ones_like(x))
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
    topSlice,
    bottomSlice,
    boundary,
    GFr,
):
    topSlice = 10**topSlice
    bottomSlice = 10**bottomSlice
    GFr = [1, GFr]
    topMask = (jnp.arange(topSlice.shape[-2])[None, None, :, None] < boundary).astype(
        jnp.float32
    )
    bottomMask = (
        jnp.arange(topSlice.shape[-2])[None, None, :, None] >= boundary
    ).astype(jnp.float32)
    n, c, m, n = topSlice.shape
    GF = GuidedFilter(r=GFr, eps=1)

    result0, num0 = GF(bottomSlice, bottomMask)
    result1, num1 = GF(topSlice, topMask)

    num0 = num0 == (2 * GFr[1] + 1) * (2 * GFr[1] + 1) * GFr[0]
    num1 = num1 == (2 * GFr[1] + 1) * (2 * GFr[1] + 1) * GFr[0]

    result0 = result0.at[num0].set(1)
    result1 = result1.at[num1].set(1)
    result0 = result0.at[num1].set(0)
    result1 = result1.at[num0].set(0)

    t = result0 + result1

    result0, result1 = result0 / t, result1 / t

    minn, maxx = min(topSlice.min(), bottomSlice.min()), max(
        topSlice.max(), bottomSlice.max()
    )

    bottom_seg = (
        result0 * bottomSlice[:, c // 2 : c // 2 + 1, :, :]
    )  # + result0detail * bottomDetail
    top_seg = (
        result1 * topSlice[:, c // 2 : c // 2 + 1, :, :]
    )  # + result1detail * topDetail

    result = jnp.clip(bottom_seg + top_seg, minn, maxx)
    bottom_seg = jnp.clip(bottom_seg, bottomSlice.min(), bottomSlice.max())
    top_seg = jnp.clip(top_seg, topSlice.min(), topSlice.max())
    return jnp.log10(result), jnp.stack((result1[0, 0], result0[0, 0]), 0)[None]
