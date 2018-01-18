import cv2
import numpy as np

img = cv2.imread("/Users/tufengzhi/Documents/RA/toy/MNIST/result/Model1/occl/12_occl_3_7_8.png", cv2.IMREAD_GRAYSCALE)

img2 = cv2.blur(img, (3, 3))

# hist,bins = np.histogram(img.flatten(),256,[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()
# cdf_m = np.ma.masked_equal(cdf,0)
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# cdf = np.ma.filled(cdf_m,0).astype('uint8')
# img2 = cdf[img]

cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img2)

cv2.waitKey(0)