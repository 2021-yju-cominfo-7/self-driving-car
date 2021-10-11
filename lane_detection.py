import cv2
import numpy as np
import math

_RED = (255, 0, 0)
_GREEN = (0, 255, 0)
_BLUE = (0, 0, 255)
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)


def find_lane(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    mid_point = np.int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:mid_point])
    right_base = np.argmax(histogram[mid_point:]) + mid_point

    return left_base, right_base


def get_lane_slope(image, left_current, right_current):
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

    ret = {'left_fitx': ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret


def draw_lane_lines(warped_image, minv, draw_info):
    left_fitx, right_fitx, ploty = draw_info['left_fitx'], draw_info['right_fitx'], draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), _WHITE)

    return pts_mean, color_warp


def get_direction_slope(pts_mean, color_warp):
    center = np.squeeze(np.int_([pts_mean]))
    start, end = center[-1], center[0]
    arr = [start[0], start[1], end[0], end[1]]

    # 방향 각도 계산
    rad = math.atan2(arr[3] - arr[1], arr[2] - arr[0])
    deg = int((rad * 180) / math.pi - 90)

    mid_idx = int(len(np.squeeze(pts_mean)) / 2)
    # 곡률 거리 계산
    mid1 = np.int_([(start[0] + end[0]) / 2, (start[1] + end[1]) / 2])
    mid2 = np.squeeze(np.int_([pts_mean]))[mid_idx]

    x = (mid1[0] - mid2[0]) ** 2
    y = (mid1[1] - mid2[1]) ** 2
    dist = int((x + y) ** 0.5)

    cv2.circle(color_warp, mid1, 10, _BLACK, -1)
    cv2.circle(color_warp, mid2, 10, _GREEN, -1)
    cv2.circle(color_warp, start, 10, _RED, -1)
    cv2.circle(color_warp, end, 10, _BLACK, -1)

    return deg, dist, color_warp


def add_img_weighted(original_image, color_warp, minv):
    new_warp = cv2.warpPerspective(color_warp, minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, new_warp, 0.4, 0)

    return result
