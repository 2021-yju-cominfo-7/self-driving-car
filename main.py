import cv2
from image_processing import set_img_marker, make_wrapping_img, make_filtering_img, set_roi_area
from lane_detection import find_lane, get_lane_slope, draw_lane_lines, get_direction_slope, add_img_weighted
from serial_arduino import make_serial_connection, write_signal, check_order


def make_image(image):
    # MEMO 해상도에 따라 이미지 리사이징 필요
    # image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    (h, w) = (image.shape[0], image.shape[1])

    mark_img, src_position = set_img_marker(image)
    wrap_img, minv = make_wrapping_img(image, src_position)
    filter_img = make_filtering_img(wrap_img)
    roi_img = set_roi_area(filter_img)

    return minv, roi_img


def get_lane_information(original_image, roi_image, minv, correction):
    _DEG_ERROR_RANGE = 1
    _DIST_ERROR_RANGE = 3

    left, right = find_lane(roi_image)
    draw_info = get_lane_slope(roi_image, left, right)
    pts_mean, color_warp = draw_lane_lines(roi_image, minv, draw_info)
    deg, dist, color_warp = get_direction_slope(pts_mean, color_warp)
    result = add_img_weighted(original_image, color_warp, minv)

    flag = dist > _DIST_ERROR_RANGE
    deg = (deg if flag else 0) + correction

    direction = "LEFT" if ((deg < _DEG_ERROR_RANGE * -1) and flag) \
        else ("RIGHT" if ((deg > _DEG_ERROR_RANGE) and flag)
              else "FRONT")

    left = int(original_image.shape[1] * 0.03)
    top = int(original_image.shape[0] * 0.1)

    cv2.putText(result, f"[{direction}]", (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Deg : {deg}", (left, top + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Dist : {dist}", (left, top + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result, flag, deg


def main():
    speed = 0
    correction = 0

    # TODO 차량 연결 시, 활성화
    # connection = make_serial_connection()
    # MEMO 웹캠 정보 가져오기
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./video/ex3.mp4")

    while True:
        # TODO 프레임 값 수정 필요
        ret, img = cap.read()
        key = cv2.waitKey(30)

        if not ret:
            break

        if key == 32:  # CAR START or STOP
            speed = 9 if speed < 9 else 0
        elif key == 119:  # SPEED UP
            if speed == 0:
                speed = 9
            else:
                speed += 1
        elif key == 115:  # SPEED DOWN
            speed -= 1
        elif key == 97:
            correction -= 1
        elif key == 100:
            correction += 1

        if speed < 9:
            speed = 0

        minv, roi_img = make_image(img)

        try:
            result, flag, deg = get_lane_information(img, roi_img, minv, correction)
            order_msg = check_order(speed, deg)

            # TODO 차량 연결 시, 활성화
            # write_signal(connection, speed, deg)
        except:
            result = img

        left = int(result.shape[1] * 0.03)
        top = int(result.shape[0] * 0.7)

        cv2.putText(result, f"{order_msg}", (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, f"Throttles : {speed}", (left, top + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, f"Deg_corr : {correction}", (left, top + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("result", result)

        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
