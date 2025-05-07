# Feature extraction (HOG, CNN)
from skimage.feature import hog
from preprocessing import resize_image
def feature_extractor(image):
  resized=resize_image(image,(64,64))
  features = hog(resized,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)

  return features