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
        (w * 0.02, h * 0.9), (w * 0.22, h * 0.2), (w * 0.98, h * 0.9), (w * 0.78, h * 0.2)
    ])
    position = position.astype(int)

    cv2.circle(marked_img, position[_LB], marker_size, _RED, -1)
    cv2.circle(marked_img, position[_LT], marker_size, _GREEN, -1)
    cv2.circle(marked_img, position[_RB], marker_size, _BLUE, -1)
    cv2.circle(marked_img, position[_RT], marker_size, _BLACK, -1)

    return marked_img, position


def make_wrapping_img(image, source_position):
    (h, w) = (image.shape[0], image.shape[1])
    source = np.float32(source_position)

    destination_position = [(w * 0.15, h * 0.95), (w * 0.15, h * 0.15), (w * 0.85, h * 0.95), (w * 0.85, h * 0.15)]
    destination = np.float32(destination_position)

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minverse = cv2.getPerspectiveTransform(destination, source)
    wrapped_img = cv2.warpPerspective(image, transform_matrix, (w, h))

    return wrapped_img, minverse


def make_filtering_img(image):
    g_blur_size = 15
    m_blur_size = 15
    thresh = 190

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_blur_img = cv2.GaussianBlur(gray_img, (g_blur_size, g_blur_size), 0)
    m_blur_img = cv2.medianBlur(g_blur_img, m_blur_size)
    ret, thr_img = cv2.threshold(m_blur_img, thresh, 255, cv2.THRESH_BINARY)
    cv2.imshow("roi", set_roi_area(m_blur_img))
    # cv2.imshow("test", m_blur_img)

    canny_img = cv2.Canny(thr_img, 30, 350)

    filtered_img = canny_img

    return filtered_img


def set_roi_area(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array([
        [int(0.1 * x), int(0.9 * y)], [int(0.1 * x), int(0.1 * y)],
        [int(0.49 * x), int(0.1 * y)], [int(0.49 * x), int(0.9 * y)],
        [int(0.51 * x), int(0.9 * y)], [int(0.51 * x), int(0.1 * y)],
        [int(0.9 * x), int(0.1 * y)], [int(0.9 * x), int(0.9 * y)],
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
