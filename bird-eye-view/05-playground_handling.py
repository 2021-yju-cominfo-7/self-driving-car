import math

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

_LB = 0
_LT = 1
_RB = 2
_RT = 3

_RED = (255, 0, 0)
_GREEN = (0, 255, 0)
_BLUE = (0, 0, 255)
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)

_PAUSE_TIME = 0.01
_DEG_ERROR_RANGE = 1
_DIST_ERROR_RANGE = 8


def make_source_marker(image):
    marked_img = image.copy()
    marker_size = 10
    # lb, lt, rb, rt
    position = [
        (60, 350), (170, 80), (630, 350), (510, 80)
    ]
    cv2.circle(marked_img, position[_LB], marker_size, _RED, -1)
    cv2.circle(marked_img, position[_LT], marker_size, _GREEN, -1)
    cv2.circle(marked_img, position[_RB], marker_size, _BLUE, -1)
    cv2.circle(marked_img, position[_RT], marker_size, _BLACK, -1)

    return marked_img, position


def wrapping_img(image, source_position):
    (h, w) = (image.shape[0], image.shape[1])

    source = np.float32(source_position)
    destination = np.float32([(60, 380), (60, 50), (630, 380), (630, 50)])

    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minverse = cv2.getPerspectiveTransform(destination, source)
    wrapped_img = cv2.warpPerspective(image, transform_matrix, (w, h))

    return wrapped_img, minverse


def color_filtering_img(image):
    g_blur_size = 7
    m_blur_size = 13
    thresh = 170

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_blur_img = cv2.GaussianBlur(gray_img, (g_blur_size, g_blur_size), 0)
    m_blur_img = cv2.medianBlur(g_blur_img, m_blur_size)
    ret, thr_img = cv2.threshold(m_blur_img, thresh, 255, cv2.THRESH_BINARY)
    canny_img = cv2.Canny(thr_img, 30, 350)

    filtered_img = canny_img

    return filtered_img


def set_roi_area(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array([
        [int(0.02 * x), int(0.9 * y)], [int(0.02 * x), int(0.1 * y)],
        [int(0.4 * x), int(0.1 * y)], [int(0.4 * x), int(0.9 * y)],
        [int(0.6 * x), int(0.9 * y)], [int(0.6 * x), int(0.1 * y)],
        [int(0.98 * x), int(0.1 * y)], [int(0.98 * x), int(0.9 * y)],
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


def plot_histogram(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    mid_point = np.int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:mid_point])
    right_base = np.argmax(histogram[mid_point:]) + mid_point

    return left_base, right_base


def slide_window_search(image, left_current, right_current):
    out_img = np.dstack((image, image, image))

    nwindows = 4
    window_height = np.int(image.shape[0] / nwindows)
    nonzero = image.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    thickness = 2

    for w in range(nwindows):
        win_y_low = image.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = image.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), _GREEN, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), _GREEN, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)

        if len(good_left) > minpix:
            left_current = np.int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = _RED
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = _BLUE

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='red')
    # plt.plot(right_fitx, ploty, color='blue')
    # plt.xlim(0, 720)
    # plt.ylim(440, 0)
    # plt.show(block=False)
    # plt.pause(_PAUSE_TIME)
    # plt.cla()

    ret = {'left_fitx': ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret


def draw_lane_lines(original_image, warped_image, minv, draw_info):
    left_fitx, right_fitx, ploty = draw_info['left_fitx'], draw_info['right_fitx'], draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    left_line = {"start": np.int_(pts_left[0][0]), "end": np.int_(pts_left[0][-1])}
    cv2.line(color_warp, left_line["start"], left_line["end"], (0, 255, 0), 10)

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), _WHITE)

    # todo test code
    center = np.squeeze(np.int_([pts_mean]))
    start, end = center[-1], center[0]
    arr = [start[0], start[1], end[0], end[1]]

    # 방향 각도 계산
    rad = math.atan2(arr[3] - arr[1], arr[2] - arr[0])
    deg = int((rad * 180) / math.pi - 90)

    # 곡률 거리 계산
    mid1 = np.int_([(start[0] + end[0]) / 2, (start[1] + end[1]) / 2])
    mid2 = np.squeeze(np.int_([pts_mean]))[202]

    x = (mid1[0] - mid2[0]) ** 2
    y = (mid1[1] - mid2[1]) ** 2
    dist = int((x + y) ** 0.5)

    cv2.circle(color_warp, mid1, 5, _BLACK, -1)
    cv2.circle(color_warp, mid2, 5, _GREEN, -1)
    cv2.circle(color_warp, start, 10, _RED, -1)
    cv2.circle(color_warp, end, 10, _BLACK, -1)

    # cv2.imshow("color_warp", color_warp)

    new_warp = cv2.warpPerspective(color_warp, minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 0.5, new_warp, 1, 0)

    return pts_mean, result, deg, dist


cap = cv2.VideoCapture("../video/ex3.mp4")
winname = "result"

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('result.avi', fourcc, 30.0, (int(width), int(height)))

cv2.namedWindow(winname)
# cv2.moveWindow(winname, 100, 0)

while True:
    ret, img = cap.read()

    if not ret:
        break

    (h, w) = (img.shape[0], img.shape[1])

    mark_img, src_position = make_source_marker(img)
    wrap_img, minv = wrapping_img(img, src_position)
    filter_img = color_filtering_img(wrap_img)
    roi_img = set_roi_area(filter_img)

    left, right = plot_histogram(roi_img)
    draw_info = slide_window_search(roi_img, left, right)
    mean_pts, result, deg, dist = draw_lane_lines(img, roi_img, minv, draw_info)
    dir = "LEFT" if ((deg < _DEG_ERROR_RANGE * -1) and dist > _DIST_ERROR_RANGE) \
        else ("RIGHT" if ((deg > _DEG_ERROR_RANGE) and dist > _DIST_ERROR_RANGE)
              else "FRONT")

    cv2.putText(result, f"Deg : {deg}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(result, f"Dist : {dist}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(result, f"[{dir}]", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.imshow("mark", mark_img)
    cv2.imshow("roi", set_roi_area(wrap_img))
    cv2.imshow(winname, result)
    out.write(result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
