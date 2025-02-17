import numpy as np

image_small = np.load(
    "C:/Users/yu/OneDrive - TUM/basicpy/BaSiCPy/3DLightsheetfromAli/mouse_brain_raw_scans/CD1-41_Z1000-1009/images_small.npy"
)
th = 300
image2 = np.reshape(
    image_small,
    [
        image_small.shape[0],
        image_small.shape[1] * image_small.shape[2],
        *image_small.shape[3:],
    ],
)
c_channelno = 1
weights = image2 > th
temp_data = np.reshape(
    np.squeeze(image2[c_channelno, :, :, :, :]),
    (-1, np.shape(image2)[3], np.shape(image2)[4]),
)
temp_mask = np.reshape(
    np.squeeze(weights[c_channelno, :, :, :, :]),
    (-1, np.shape(image2)[3], np.shape(image2)[4]),
)

from basicpy import BaSiC

b_seg_approx_autotune = BaSiC(
    get_darkfield=False,
    smoothness_flatfield=8.0,
    smoothness_darkfield=0.8,
    fitting_mode="approximate",
    sort_intensity=False,
)

b_seg_approx_autotune.fit(temp_data, fitting_weight=temp_mask)
print(b_seg_approx_autotune.darkfield.min(), b_seg_approx_autotune.darkfield.max())
import matplotlib.pyplot as plt

plt.imshow(b_seg_approx_autotune.flatfield, cmap="gray")
plt.colorbar()
plt.show()
plt.imshow(b_seg_approx_autotune.darkfield, cmap="gray")
plt.colorbar()
plt.show()
