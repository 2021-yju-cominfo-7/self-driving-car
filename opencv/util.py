import cv2
import numpy as np
from image_processing import region_of_interest, hough_lines


def get_line(img):
    left_roi = region_of_interest(img, img.shape[:2], "left")
    right_roi = region_of_interest(img, img.shape[:2], "right")

    L_lines = np.squeeze(hough_lines(left_roi, 30, 10, 20))[:, None]
    R_lines = np.squeeze(hough_lines(right_roi, 30, 10, 20))[:, None]

    return L_lines, R_lines


def get_fitline(img, f_lines):
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)

    rows, cols = img.shape[:2]
    # print(rows, cols)
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]

    # leftY = int((-x * vy / vx) + y)
    # rightY = int(((cols - x) * vy / vx) + y)
    #
    # x1, y1 = cols - 1, rightY
    # x2, y2 = 0, leftY

    # x1, y1 = int(x[0] - 50), int(y[0] - (50 * vy / vx))
    # x2, y2 = int((x + 50)[0]), int((y + (50 * vy / vx))[0])

    # print(int(np.squeeze(x - 50)), int(np.squeeze(y - (50 * vy / vx))))
    # print(int(np.squeeze(x + 50)), int(np.squeeze(y + (50 * vy / vx))))
    x1, y1 = int(np.squeeze(x - 50)), int(np.squeeze(y - (50 * vy / vx)))
    x2, y2 = int(np.squeeze(x + 50)), int(np.squeeze(y + (50 * vy / vx)))

    result = [x1, y1, x2, y2]

    return result


def draw_lines(img, lines, color=[0, 0, 255], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=5):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def weighted_img(img, initial_img, α=0.3, β=1., λ=1.):
    return cv2.addWeighted(initial_img, α, img, β, λ)
