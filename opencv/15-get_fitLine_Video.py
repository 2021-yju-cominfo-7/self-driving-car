import cv2
import numpy as np


def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


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


def draw_lines(img, lines, color=[0, 0, 255], thickness=5):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=5):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    return lines


def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    # return cv2.addWeighted(initial_img, α, img, β, λ)
    return cv2.addWeighted(initial_img, 0.3, img, 1, 1)


def get_fitline(img, f_lines):
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)

    rows, cols = img.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]

    leftY = int((-x * vy / vx) + y)
    rightY = int(((cols - x) * vy / vx) + y)

    x1, y1 = cols - 1, rightY
    x2, y2 = 0, leftY

    result = [x1, y1, x2, y2]

    return result


cap = cv2.VideoCapture("../video/ex3.mp4")

# 재생할 파일의 넓이와 높이
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(width), int(height)))

while (cap.isOpened()):
    ret, image = cap.read()

    if image is None:
        break

    height, width = image.shape[:2]
    try:

        gray_img = grayscale(image)
        blur_img = gaussian_blur(gray_img, 5)

        # 미디안 블러링으로 좌우의 배수구망 이미지 제거
        blur_img = cv2.medianBlur(blur_img, 11)
        ret, thr_img = cv2.threshold(blur_img, 170, 255, cv2.THRESH_BINARY)

        # canny_img = canny(blur_img, 30, 350)
        canny_img_thr = canny(thr_img, 30, 350)

        # TODO 이미지 프로세싱 결과 점검
        # cv2.imshow("image_processing1", blur_img)
        # cv2.imshow("image_processing2", thr_img)

        vertices = np.array(
            [[(0, height / 2 + 100), (width / 2 - 200, 70), (width / 2 + 200, 70), (width, height / 2 + 100)]],
            dtype=np.int32)

        # TODO ROI 영역 설정 결과 점검
        check_roi = region_of_interest(image, vertices)
        cv2.imshow("ROI", check_roi)

        ROI_img = region_of_interest(canny_img_thr, vertices)
        line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)
        # line_arr = hough_lines(canny_img, 1, 1 * np.pi / 180, 30, 10, 20)
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
        temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        L_lines, R_lines = L_lines[:, None], R_lines[:, None]

        # 왼쪽, 오른쪽 각각 대표선 구하기
        left_fit_line = get_fitline(image, L_lines)
        right_fit_line = get_fitline(image, R_lines)

        draw_lines(temp, L_lines, [0, 0, 255])
        draw_lines(temp, R_lines, [255, 0, 0])

        # 대표선 그리기
        draw_fit_line(temp, left_fit_line, [0, 255, 0])
        draw_fit_line(temp, right_fit_line, [0, 255, 0])

        result = weighted_img(temp, image)
    except:
        result = image

    cv2.imshow('result', result)
    out.write(result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
