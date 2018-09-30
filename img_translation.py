# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage.transform import warp, SimilarityTransform
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft

ref_image_fn    = 'RgsASPc_A.jpg'
def_image_fn    = 'RgsASPc_C.jpg'
output_image_fn = 'alianed.png'
pixel_precision = 100

ref_image = skimage.io.imread(ref_image_fn)
def_image = skimage.io.imread(def_image_fn)

ref_image_gray = skimage.color.rgb2gray(ref_image)
def_image_gray = skimage.color.rgb2gray(def_image)

image_product = np.fft.fft2(ref_image_gray) * np.fft.fft2(def_image_gray).conj()

# subpixel precision
shift, error, diffphase = register_translation(ref_image_gray, def_image_gray, pixel_precision)

tform1 = SimilarityTransform(translation=(-shift[1], -shift[0]))
alianed = warp(def_image, tform1, mode='symmetric')

fig = plt.figure(figsize=(20, 10))
ax1 = plt.subplot(1, 4, 1)
ax2 = plt.subplot(1, 4, 2)
ax3 = plt.subplot(1, 4, 3)
ax4 = plt.subplot(1, 4, 4)

ax1.imshow(ref_image)
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(def_image)
ax2.set_axis_off()
ax2.set_title('Defect image')

ax3.imshow(alianed)
ax3.set_axis_off()
ax3.set_title('Alianment')

cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
ax4.imshow(cc_image.real)
ax4.set_axis_off()
ax4.set_title("Supersampled XC sub-area")

plt.show()

skimage.io.imsave(output_image_fn, alianed)

print("Detected subpixel offset (y, x): {}".format(shift))
