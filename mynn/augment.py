import numpy as np


def random_shift(X, max_shift=2):
    """
    Random translation per image, empty regions filled with 0.
    X: [N, C, H, W]
    """
    N, C, H, W = X.shape
    shifted = np.zeros_like(X)
    for n in range(N):
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)
        sx1, sx2 = max(0, -dx), min(H, H - dx)
        sy1, sy2 = max(0, -dy), min(W, W - dy)
        tx1, tx2 = max(0, dx), min(H, H + dx)
        ty1, ty2 = max(0, dy), min(W, W + dy)
        shifted[n, :, tx1:tx2, ty1:ty2] = X[n, :, sx1:sx2, sy1:sy2]
    return shifted


def random_noise(X, std=0.05):
    """
    Add Gaussian noise to each pixel, clamped to [0, 1].
    X: [N, C, H, W]
    """
    noise = np.random.randn(*X.shape).astype(X.dtype) * std
    return np.clip(X + noise, 0.0, 1.0)


def random_horizontal_flip(X, p=0.5):
    """
    Random horizontal flip with probability p.
    X: [N, C, H, W]
    """
    mask = np.random.random(X.shape[0]) < p
    flipped = X.copy()
    flipped[mask] = X[mask][:, :, :, ::-1]
    return flipped


def _bilinear_sample(img, x_src, y_src):
    """
    Bilinear interpolation: sample img at fractional coordinates (x_src, y_src).
    img: [H, W], x_src/y_src: [H, W] arrays of source coordinates.
    Returns sampled array [H, W], 0 where out of bounds.
    """
    H, W = img.shape
    x0 = np.floor(x_src).astype(int)
    y0 = np.floor(y_src).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    x0, x1 = np.clip(x0, 0, W - 1), np.clip(x1, 0, W - 1)
    y0, y1 = np.clip(y0, 0, H - 1), np.clip(y1, 0, H - 1)

    wx = x_src - x0.astype(np.float64)
    wy = y_src - y0.astype(np.float64)

    result = (img[y0, x0] * (1 - wx) * (1 - wy) +
              img[y0, x1] * wx * (1 - wy) +
              img[y1, x0] * (1 - wx) * wy +
              img[y1, x1] * wx * wy)
    return result * valid


def random_rotation(X, max_deg=10):
    """
    Random rotation per image by angle in [-max_deg, max_deg].
    Uses bilinear interpolation, empty regions filled with 0.
    X: [N, C, H, W]
    """
    N, C, H, W = X.shape
    rotated = np.zeros_like(X)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float64)
    x_centered = x_grid - cx
    y_centered = y_grid - cy

    for n in range(N):
        angle = np.random.uniform(-max_deg, max_deg)
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_src = cos_t * x_centered + sin_t * y_centered + cx
        y_src = -sin_t * x_centered + cos_t * y_centered + cy

        for c in range(C):
            rotated[n, c] = _bilinear_sample(X[n, c], x_src, y_src)
    return rotated


def random_resize(X, min_scale=0.9, max_scale=1.1):
    """
    Random resizing per image by a factor in [min_scale, max_scale].
    Center-crops if enlarged, zero-pads if shrunk.
    X: [N, C, H, W]
    """
    N, C, H, W = X.shape
    resized = np.zeros_like(X)
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float64)

    for n in range(N):
        scale = np.random.uniform(min_scale, max_scale)
        # Map output coords to source coords: src = (out - center) / scale + center
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        x_src = (x_grid - cx) / scale + cx
        y_src = (y_grid - cy) / scale + cy

        for c in range(C):
            resized[n, c] = _bilinear_sample(X[n, c], x_src, y_src)
    return resized


def compose_augment(X, shift=0, noise=0.0, hflip=False, rotate=0, resize=0.0):
    """
    Apply multiple augmentations in sequence. Handles both 2D [N, D] (MLP)
    and 4D [N, C, H, W] (CNN) inputs by reshaping 2D to 4D and back.
    shift:  max pixels for random translation (0 = disabled)
    noise:  std of Gaussian noise (0 = disabled)
    hflip:  enable random horizontal flip
    rotate: max degrees for random rotation (0 = disabled)
    resize: max relative scale change, e.g. 0.1 means [0.9, 1.1] (0 = disabled)
    """
    is_2d = X.ndim == 2
    if is_2d:
        N, D = X.shape
        H = W = int(np.sqrt(D))
        X = X.reshape(N, 1, H, W)

    if rotate > 0:
        X = random_rotation(X, max_deg=rotate)
    if shift > 0:
        X = random_shift(X, max_shift=shift)
    if resize > 0:
        X = random_resize(X, min_scale=1.0 - resize, max_scale=1.0 + resize)
    if noise > 0:
        X = random_noise(X, std=noise)
    if hflip:
        X = random_horizontal_flip(X)

    if is_2d:
        X = X.reshape(N, D)
    return X
