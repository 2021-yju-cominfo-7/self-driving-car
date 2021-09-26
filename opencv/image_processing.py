import cv2
import numpy as np


def hough_lines(img, threshold, min_line_len, max_line_gap, rho=1, theta=np.pi / 180):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)

    return lines


def region_of_interest(img, size, flag, color3=(255, 255, 255), color1=255):
    height, width = size

    # 좌하, 좌상, 우상, 우하
    roi_area = {
        "left": [
            (80, height / 2 + 150), (width / 2 - 190, 100), (width / 2 - 60, 100), (width / 2 - 60, height / 2 + 150)
        ],
        "right": [
            (width / 2 + 40, height / 2 + 150), (width / 2 + 40, 100), (width / 2 + 170, 100),
            (width - 100, height / 2 + 150)
        ]
    }

    vertices = np.array([roi_area[flag]], dtype=np.int32)

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = color3
    else:
        color = color1

    cv2.fillPoly(mask, vertices, color)

    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def find_edges(img, thresh, maxval, low_threshold, high_threshold, type=cv2.THRESH_BINARY):
    ret, thr = cv2.threshold(img, thresh, maxval, type)
    canny = cv2.Canny(thr, low_threshold, high_threshold)
    return canny


def color_filter(image, gaussian_size=5, median_size=11):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (gaussian_size, gaussian_size), 0)
    median = cv2.medianBlur(blur, median_size)

    return median
