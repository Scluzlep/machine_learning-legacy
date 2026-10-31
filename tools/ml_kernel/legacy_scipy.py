"""Implement removed scipy.misc image helpers with Pillow, not Notebook edits."""
from __future__ import annotations


def install():
    import numpy as np
    from PIL import Image
    import scipy.misc as misc

    if not hasattr(misc, "imread"):
        def imread(filename, flatten=False, mode=None):
            with Image.open(filename) as original:
                if mode is not None:
                    original = original.convert(mode)
                elif flatten:
                    original = original.convert("F")
                return np.asarray(original)
        misc.imread = imread

    # The original logistic-regression image exercise also calls
    # scipy.ndimage.imread (removed alongside the scipy.misc aliases).
    import scipy.ndimage as ndimage
    if not hasattr(ndimage, "imread"):
        ndimage.imread = misc.imread

    if not hasattr(misc, "imresize"):
        def imresize(array, size, interp="bilinear", mode=None):
            method = {
                "nearest": Image.Resampling.NEAREST,
                "lanczos": Image.Resampling.LANCZOS,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "cubic": Image.Resampling.BICUBIC,
            }.get(interp)
            if method is None:
                raise ValueError(f"Unknown interpolation method: {interp}")
            image = Image.fromarray(np.asarray(array), mode=mode)
            if isinstance(size, (int, float)):
                scale = float(size) / 100 if isinstance(size, int) else float(size)
                target = (max(1, round(image.width * scale)),
                          max(1, round(image.height * scale)))
            else:
                target = (int(size[1]), int(size[0]))  # SciPy is height, width
            return np.asarray(image.resize(target, resample=method))
        misc.imresize = imresize

    if not hasattr(misc, "imsave"):
        def imsave(filename, array):
            Image.fromarray(np.asarray(array)).save(filename)
        misc.imsave = imsave
