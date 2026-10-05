import glob, os, sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

SRC = sys.argv[1]
OUT = sys.argv[2]
LOGO = sys.argv[3]
os.makedirs(OUT, exist_ok=True)

logo = Image.open(LOGO).convert("RGBA")
logo = logo.crop(logo.getbbox())

# search window around the tag
X0, X1, Y0, Y1 = 236, 316, 556, 596


def fit_surface(img, valid, box):
    """Fit a quadratic surface per channel to valid pixels inside box."""
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    xn, yn = (xx - x0) / (x1 - x0), (yy - y0) / (y1 - y0)
    A = np.stack([np.ones_like(xn), xn, yn, xn * yn, xn**2, yn**2], -1)
    v = valid[y0:y1, x0:x1]
    out = np.empty((y1 - y0, x1 - x0, 3))
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A[v], img[y0:y1, x0:x1, c][v], rcond=None)
        out[..., c] = A @ coef
    return out


for f in sorted(glob.glob(os.path.join(SRC, "*.jpg"))):
    im = Image.open(f).convert("RGB")
    a = np.asarray(im).astype(float)
    win = a[Y0:Y1, X0:X1]
    g = win.mean(2)
    r, b = win[..., 0], win[..., 2]
    content = (g < 150) | ((r - b) > 35)
    lab, n = ndimage.label(content)
    # keep components of reasonable size (letters / icon)
    sizes = ndimage.sum(content, lab, range(1, n + 1))
    edge = set(np.unique(lab[0])) | set(np.unique(lab[:, 0])) | set(np.unique(lab[:, -1]))
    content = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 6 and i + 1 not in edge])
    ys, xs = np.nonzero(content)
    cx0, cx1 = xs.min() + X0, xs.max() + X0 + 1
    cy0, cy1 = ys.min() + Y0, ys.max() + Y0 + 1

    # mask of everything to erase: content bbox padded, refined to non-label pixels
    full = np.zeros(a.shape[:2], bool)
    full[Y0:Y1, X0:X1] = content
    full = ndimage.binary_dilation(full, iterations=3)
    box = (cx0 - 5, cy0 - 4, cx1 + 5, cy1 + 4)
    bx0, by0, bx1, by1 = box
    valid = ~full
    surf = fit_surface(a, valid, box)
    # residual noise of the label surface to keep the photo texture
    res = a[by0:by1, bx0:bx1] - surf
    sigma = res[valid[by0:by1, bx0:bx1]].std(0)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, surf.shape) * np.minimum(sigma, 2.5)
    noise = ndimage.gaussian_filter(noise, (0.6, 0.6, 0))
    fill = surf + noise
    # feathered blend of the fill over the erased region
    m = full[by0:by1, bx0:bx1].astype(float)
    m = np.clip(ndimage.gaussian_filter(m, 0.8) * 1.5, 0, 1)[..., None]
    a[by0:by1, bx0:bx1] = a[by0:by1, bx0:bx1] * (1 - m) + fill * m

    # place the Velvo logo centred on the old mark, same width as the old wordmark
    w = cx1 - cx0 + 2
    h = round(w * logo.height / logo.width)
    S = 4
    big = logo.resize((w * S, h * S), Image.LANCZOS)
    lg = big.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.35))
    la = np.asarray(lg).astype(float)
    alpha = la[..., 3:4] / 255.0
    px = round((cx0 + cx1) / 2 - w / 2)
    py = round((cy0 + cy1) / 2 - h / 2)
    region = a[py:py + h, px:px + w]
    # multiply blend so the logo picks up the tag's lighting / tint
    tint = surf.mean((0, 1)) / 245.0
    blended = la[..., :3] * tint
    a[py:py + h, px:px + w] = region * (1 - alpha) + blended * alpha

    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    out.save(os.path.join(OUT, os.path.basename(f)), quality=95, subsampling=0)
    print(os.path.basename(f), (cx0, cy0, cx1, cy1), (px, py, w, h))
