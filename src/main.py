# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:56:17 2024

@author: 15210
"""

import jax

jax.config.update("jax_platform_name", "cpu")
from basicpy.basicpy import BaSiC
import tifffile
import os
from skimage.transform import resize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import scipy.io as io

x_size = 256
y_size = 256
channel = 2

GT = tifffile.imread("D:/BaSiC/simulationstudy/simulatedimages/movie/foreground.tif")
bkg_te = tifffile.imread(
    "D:/BaSiC/simulationstudy/simulatedimages/background/FITC_20x.tif"
)[0]
bkg_te = resize(bkg_te, [x_size, y_size], order=1)
flatfield_cal = bkg_te / bkg_te.mean()
darkfield_cal = 0
base_fi = np.exp(-np.arange(1, GT.shape[0] + 1) * 0.05) * 50
fi = GT[:, :, :, channel] / 2 + base_fi[:, None, None]
fi_distorted = fi * flatfield_cal + darkfield_cal + np.random.randn(x_size, y_size)
fi_seg = tifffile.imread(
    "D:/BaSiC/simulationstudy/simulatedimages/movie/segmentation.tif"
)[:, :, :, channel]
fi_seg = skimage.morphology.binary_dilation(fi_seg > 0, np.ones((1, 11, 11)))
fi_distorted = io.loadmat("D:/BaSiC/simulationstudy/fi_distorted.mat")[
    "fi_distorted"
].transpose(2, 0, 1)


basic = BaSiC(
    get_darkfield=True,
    fitting_mode="approximate",
    sort_intensity=False,
    max_reweight_iterations=5,
    smoothness_flatfield=0.990265494633178,
    smoothness_darkfield=0.098053098926636,
)
# basic.fit(fi_distorted)
fi_distorted = fi_distorted - fi_distorted.min()
basic.fit(
    images=fi_distorted,
    fitting_weight=~fi_seg,
)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
im = axes[0].imshow(basic.flatfield)
fig.colorbar(im, ax=axes[0])
axes[0].set_title("Flatfield")
im = axes[1].imshow(basic.darkfield)
fig.colorbar(im, ax=axes[1])
axes[1].set_title("Darkfield")
axes[2].plot(basic.baseline)
axes[2].set_xlabel("Frame")
axes[2].set_ylabel("Baseline")
fig.tight_layout()
plt.show()
