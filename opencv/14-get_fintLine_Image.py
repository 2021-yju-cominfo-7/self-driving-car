import cv2  # opencv 사용
import numpy as np


def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=10):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, α=0.5, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)


def get_fitline(img, f_lines):  # 대표선 구하기
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)

    rows, cols = img.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L1, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]

    leftY = int((-x * vy / vx) + y)
    rightY = int(((cols - x) * vy / vx) + y)

    x1, y1 = cols - 1, rightY
    x2, y2 = 0, leftY

    result = [x1, y1, x2, y2]

    return result


image = cv2.imread("../img/playground/pg_center.jpeg")

height, width = image.shape[:2]  # 이미지 높이, 너비

gray_img = grayscale(image)  # 흑백이미지로 변환

blur_img = gaussian_blur(gray_img, 3)  # Blur 효과

canny_img = canny(blur_img, 70, 210)  # Canny edge 알고리즘

vertices = np.array(
    [[(0, height / 2 + 300), (width / 2 - 100, 800), (width / 2 + 100, 800), (width - 560, height / 2 + 300)]],
    dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices)  # ROI 설정

line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 300, 150, 200)  # 허프 변환
line_arr = np.squeeze(line_arr)

# 기울기 구하기
slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

# 수평 기울기 제한
line_arr = line_arr[np.abs(slope_degree) < 160]
slope_degree = slope_degree[np.abs(slope_degree) < 160]

# 수직 기울기 제한
line_arr = line_arr[np.abs(slope_degree) > 95]
slope_degree = slope_degree[np.abs(slope_degree) > 95]

# 필터링된 직선 버리기
L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
L_lines, R_lines = L_lines[:, None], R_lines[:, None]

# 왼쪽, 오른쪽 각각 대표선 구하기
left_fit_line = get_fitline(image, L_lines)
right_fit_line = get_fitline(image, R_lines)

draw_fit_line(image, left_fit_line)
draw_fit_line(image, right_fit_line)

cv2.imshow('result', image)  # 결과 이미지 출력
cv2.waitKey(0)
