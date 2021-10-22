import cv2
import numpy as np

_LB = 0
_LT = 1
_RB = 2
_RT = 3

_RED = (255, 0, 0)
_GREEN = (0, 255, 0)
_BLUE = (0, 0, 255)
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)


def set_img_marker(image):
    marked_img = image.copy()
    marker_size = 10

    (h, w) = (image.shape[0], image.shape[1])

    # TODO 추후, 카메라 설정에 따라 mark position 값 수정 필요
    # lb, lt, rb, rt
    position = np.array([
        # (w * 0.02, h * 0.9), (w * 0.22, h * 0.2), (w * 0.98, h * 0.9), (w * 0.78, h * 0.2)
        (w * 0.05, h * 0.6), (w * 0.3, h * 0.3), (w * 0.9, h * 0.6), (w * 0.68, h * 0.3)
    ])
    # MEMO 영상 테스트용
    # position = np.array([
    #     (w * 0.1, h * 0.9), (w * 0.25, h * 0.2), (w * 0.9, h * 0.9), (w * 0.75, h * 0.2)
    # ])
    position = position.astype(int)

    cv2.circle(marked_img, position[_LB], marker_size, _RED, -1)
    cv2.circle(marked_img, position[_LT], marker_size, _GREEN, -1)
    cv2.circle(marked_img, position[_RB], marker_size, _BLUE, -1)
    cv2.circle(marked_img, position[_RT], marker_size, _WHITE, -1)

    return marked_img, position


def make_wrapping_img(image, source_position):
    (h, w) = (image.shape[0], image.shape[1])
    source = np.float32(source_position)

    destination_position = [(w * 0.15, h * 0.95), (w * 0.15, h * 0.15), (w * 0.85, h * 0.95), (w * 0.85, h * 0.15)]
    # MEMO 영상 테스트용
    # destination_position = [(w * 0.1, h * 0.95), (w * 0.1, h * 0.15), (w * 0.9, h * 0.95), (w * 0.9, h * 0.15)]
    # destination_position = [(w * 0.1, h * 0.95), (w * 0.1, h * 0.15), (w * 0.9, h * 0.95), (w * 0.9, h * 0.15)]
    destination = np.float32(destination_position)

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minverse = cv2.getPerspectiveTransform(destination, source)
    wrapped_img = cv2.warpPerspective(image, transform_matrix, (w, h))

    return wrapped_img, minverse


def tmp(img_color):
    height, width = img_color.shape[:2]  # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # cvtColor 함수를 이용하여 hsv 색공간으로 변환
    g_blur_img = cv2.GaussianBlur(hsv, (19, 19), 0)
    m_blur_img = cv2.medianBlur(g_blur_img, 19)

    # H(Hue, 색조), S(Saturation, 채도), V(Value, 명도)
    # h, s, v = cv2.split(m_blur_img)

    # cv2.imshow("h", h)
    # cv2.imshow("s", s)
    # cv2.imshow("v", v)

    # # 흰색 선을 놓치면 최대 값을 높임
    # h = cv2.inRange(h, 80, 255)
    # result_h = cv2.bitwise_and(hsv, hsv, mask=h)
    # result_h = cv2.cvtColor(result_h, cv2.COLOR_HSV2BGR)
    #
    # cv2.imshow("result_h", result_h)
    #
    # # 흰색 선을 놓치면 최소 값을 낮춤
    # s = cv2.inRange(s, 5, 50)
    # result_s = cv2.bitwise_and(hsv, hsv, mask=s)
    # result_s = cv2.cvtColor(result_s, cv2.COLOR_HSV2BGR)
    #
    # cv2.imshow("result_s", result_s)
    #
    # # 흰색 선을 놓치면 최소 값을 낮춤
    # v = cv2.inRange(v, 180, 255)
    # result_v = cv2.bitwise_and(hsv, hsv, mask=v)
    # result_v = cv2.cvtColor(result_v, cv2.COLOR_HSV2BGR)
    #
    # cv2.imshow("result_v", result_v)

    img_mask = cv2.inRange(m_blur_img, (80, 5, 140), (255, 50, 255))  # 범위내의 픽셀들은 흰색, 나머지 검은색
    img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)
    # cv2.imshow('img_mask', img_mask)
    # cv2.imshow('img_color', img_result)

    gray_img = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
    ret, thr_img = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY)
    canny_img = cv2.Canny(thr_img, 30, 350)

    return canny_img


def make_filtering_img(image):
    g_blur_size = 15
    m_blur_size = 31
    thresh = 160

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_blur_img = cv2.GaussianBlur(gray_img, (g_blur_size, g_blur_size), 0)
    m_blur_img = cv2.medianBlur(g_blur_img, m_blur_size)
    ret, thr_img = cv2.threshold(m_blur_img, thresh, 255, cv2.THRESH_BINARY)
    # cv2.imshow("roi", set_roi_area(m_blur_img))

    canny_img = cv2.Canny(thr_img, 30, 350)

    filtered_img = canny_img

    return filtered_img


def set_roi_area(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array([
        [int(0.05 * x), int(0.9 * y)], [int(0.05 * x), int(0.1 * y)],
        [int(0.45 * x), int(0.1 * y)], [int(0.45 * x), int(0.9 * y)],
        [int(0.55 * x), int(0.9 * y)], [int(0.55 * x), int(0.1 * y)],
        [int(0.95 * x), int(0.1 * y)], [int(0.95 * x), int(0.9 * y)],
        [int(0.05 * x), int(0.9 * y)]
    ])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_roi_image = cv2.bitwise_and(image, mask)

    return masked_roi_image
