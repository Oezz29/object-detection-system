# Segmentation logic 
import numpy as np
import cv2
import matplotlib.pyplot as plt
def extract_bounded(image, min_area=1000, margin=15):
    edges = cv2.Canny(image, 30, 100)
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    image_with_boxes = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)

            boxes.append((x1, y1, x2, y2))
    return boxes


def extract_rois(image, coordinates):
    rois = []
    for (x1, y1, x2, y2) in coordinates:
        roi = image[y1:y2, x1:x2]
        rois.append(roi)
    return rois


def draw_roi_boxes(image, coordinates, color=(0, 255, 0), thickness=2):
    image_with_boxes = image.copy()
    for (x1, y1, x2, y2) in coordinates:
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)
    return image_with_boxes
