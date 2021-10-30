import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt

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

    # out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    # out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]
    #
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='red')
    # plt.plot(right_fitx, ploty, color='blue')
    # plt.xlim(0, 720)
    # plt.ylim(440, 0)
    # plt.show(block=False)
    # plt.pause(0.001)
    # plt.cla()

    ret = {'left_fitx': ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret


def draw_lane_lines(wraped_image, minv, draw_info):
    left_fitx, right_fitx, ploty = draw_info['left_fitx'], draw_info['right_fitx'], draw_info['ploty']

    warp_zero = np.zeros_like(wraped_image).astype(np.uint8)
    color_wrap = np.dstack((warp_zero, warp_zero, warp_zero))
    (h, w) = (color_wrap.shape[0], color_wrap.shape[1])

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    left_mid_idx = int(len(np.squeeze(pts_left)) / 2)
    right_mid_idx = int(len(np.squeeze(pts_right)) / 2)

    left_line = {
        "start": np.int_(pts_left[0][0]),
        "mid": np.int_(pts_left[0][left_mid_idx]),
        "end": np.int_(pts_left[0][-1])
    }
    right_line = {
        "start": np.int_(pts_right[0][0]),
        "mid": np.int_(pts_right[0][right_mid_idx]),
        "end": np.int_(pts_right[0][-1])
    }

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

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

    cv2.fillPoly(color_wrap, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_wrap, np.int_([pts_mean]), _WHITE)

    cv2.line(color_wrap, left_line["start"], left_line["end"], _BLUE, 20)
    cv2.line(color_wrap, right_line["start"], right_line["end"], _BLUE, 20)

    cv2.circle(color_wrap, left_line["mid"], 10, _GREEN, -1)
    cv2.circle(color_wrap, right_line["mid"], 10, _GREEN, -1)
    cv2.circle(color_wrap, (int(w / 2), int(h / 2)), 10, _RED, 20)

    cv2.circle(color_wrap, start, 20, _RED, -1)
    cv2.circle(color_wrap, end, 20, _BLACK, -1)
    cv2.circle(color_wrap, mid1, 10, _BLACK, -1)
    cv2.circle(color_wrap, mid2, 10, _GREEN, -1)

    cv2.imshow("wrap", color_wrap)

    check_start_line = ((right_line["start"][0] - left_line["start"][0]) < 0) or \
                       (left_line["start"][0] > start[0]) or \
                       (right_line["start"][0] < start[0])
    check_end_line = (right_line["end"][0] - left_line["end"][0]) < 0 or \
                     (left_line["end"][0] > end[0]) or \
                     (right_line["end"][0] < end[0])

    if check_start_line or check_end_line:
        now = time.localtime()
        # print(f"[%02d:%02d:%02d] 차선 인식 결과 에러!" % (now.tm_hour, now.tm_min, now.tm_sec))
        raise Exception("LINE_ERR")

    deviation = (right_line["mid"][0] - w / 2) - (w / 2 - left_line["mid"][0])

    return deg, dist, deviation, color_wrap


def add_img_weighted(original_image, color_warp, minv):
    new_warp = cv2.warpPerspective(color_warp, minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 0.5, new_warp, 1, 0)

    return result
