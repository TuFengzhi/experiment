import cv2
import numpy as np

def filter1(img):
	img = img[i].numpy()
	for i in range(img.size()[0]):
		img2 = img[i]
		hist, bins = np.histogram(img2.flatten(), 256, [0,256])
		cdf = hist.cumsum()
		cdf_normalized = cdf * hist.max() / cdf.max()
		cdf_m = np.ma.masked_equal(cdf, 0)
		cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
		cdf = np.ma.filled(cdf_m, 0).astype('uint8')
		img[i] = cdf[img2]
	img = torch.from_numpy(img)
	return img

def filter2(img, size):
	img2 = cv2.blur(img, (size, size))
	return img2

# img = cv2.imread('')
# img2 = filter2(img, 5)
# cv2.imshow("Image-1", img)
# cv2.imshow("Image-2", img2)
# cv2.waitKey(0)