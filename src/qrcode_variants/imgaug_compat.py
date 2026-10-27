import numpy as np


def ensure_numpy_sctypes():
  if hasattr(np, "sctypes"):
    return
  np.sctypes = {
    "int": [np.int8, np.int16, np.int32, np.int64],
    "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
    "float": [np.float16, np.float32, np.float64],
    "complex": [np.complex64, np.complex128],
    "others": [bool, object, bytes, str, np.void],
  }


ensure_numpy_sctypes()

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

__all__ = [
  "BoundingBox",
  "BoundingBoxesOnImage",
  "ensure_numpy_sctypes",
  "iaa",
]
