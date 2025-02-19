from basicpy.basicpy import BaSiC
from basicpy import datasets

from matplotlib import pyplot as plt
import numpy as np
from hyperactive import Hyperactive
import pandas as pd
from m2stitch import stitch_images


# brain_wsi_image = datasets.wsi_brain()
import scipy.io as io


def compose_image(img):
    img_composed = np.empty(
        (
            img.shape[1] * 7,
            img.shape[2] * 9,
        )
    )

    y = 0
    x = img.shape[2] * 8

    rows = []
    cols = []
    for j, im in enumerate(img):
        img_composed[y : y + img.shape[1], x : x + img.shape[2]] = im
        rows.append(y // img.shape[1])
        cols.append(x // img.shape[2])
        if (y // img.shape[2]) % 2 == 0:
            x -= img.shape[2]
            if x < 0:
                x = 0
                y += img.shape[1]
        else:
            x += img.shape[2]
            if x > img.shape[2] * 8:
                x = img.shape[2] * 8
                y += img.shape[1]
    return img_composed, rows, cols


x = io.loadmat(
    "C:/Users/yu/OneDrive - TUM/basicpy/BaSiC-marrlab/BaSiC-master/Demoexamples/WSI_Brain/matlab_result.mat"
)["IF_corr"]
composed_o, rows, cols = compose_image(x.transpose(2, 0, 1))

# plt.imshow(composed, vmin=970, vmax=10000, cmap="gray")
# plt.colorbar()
# plt.show()

x = io.loadmat(
    "C:/Users/yu/OneDrive - TUM/basicpy/BaSiC-marrlab/BaSiC-master/Demoexamples/WSI_Brain/uncorrected.mat"
)["IF"].transpose(2, 0, 1)

basic = BaSiC(
    get_darkfield=True,
    fitting_mode="approximate",
    sort_intensity=True,
    smoothness_flatfield=2,
    smoothness_darkfield=0.2,
)

basic.autotune(x)
basic.fit(images=x)

plt.imshow(basic.flatfield, cmap="gray")
plt.colorbar()
plt.show()

plt.imshow(basic.darkfield, cmap="gray")
plt.colorbar()
plt.show()

transformed2 = basic.transform(x)

composed, rows, cols = compose_image(transformed2)

plt.subplot(1, 2, 1)
plt.imshow(composed_o, vmin=970, vmax=10000, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(composed, vmin=970, vmax=10000, cmap="gray")

plt.show()
