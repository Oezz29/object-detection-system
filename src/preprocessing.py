# Preprocessing and augmentation functions
import cv2
import numpy as np
import random

#Image Resizing
def resize_image(image, size=(224,224)):
    resized_img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_img

#Color conversion
def convert_to_grayscale(image):
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale_img

#Noise Reduction
def bilateral_filter (image):
  blurred = cv2.bilateralFilter(image, 11, 75, 75)
  return blurred

#normalization
def normalization(image):
  normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
  return normalized


#Contrast Adjustment
def enhanced_image(image):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  enhanced = clahe.apply(image)
  return enhanced

def blurring(image):
  blurred = cv2.GaussianBlur(image, (5, 5), 0)
  return blurred

def preprocessing(img):
  resized=resize_image(img)
  gray=convert_to_grayscale(resized)
  enhanced=enhanced_image(gray)
  normalized=normalization(enhanced)
  return normalized

def apply_rotation(image, angle=None):

    if angle is None:
        angle = random.uniform(-30, 30)
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

def apply_flip(image, flip_mode=None):

    if flip_mode is None:
        flip_mode = random.choice([-1, 0, 1, 2])
    
    return cv2.flip(image, flip_mode) if flip_mode != 2 else image.copy()


def augment_image(image):

    augmented = image.copy()
    
    angle = random.uniform(-30, 30)
    h, w = augmented.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(augmented, M, (w, h), flags=cv2.INTER_LINEAR)
    
    flip_1 = cv2.flip(augmented, 1)
        
    flip_2 = cv2.flip(augmented, 0)
    
    return rotated, flip_1, flip_2