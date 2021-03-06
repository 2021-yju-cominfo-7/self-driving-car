import cv2
from image_processing import set_img_marker, make_wrapping_img, make_filtering_img, tmp, set_roi_area
from lane_detection import find_lane, get_lane_slope, draw_lane_lines, add_img_weighted
from serial_arduino import make_serial_connection, write_signal, check_order
from lidar_controller import make_connection, sensing_lidar
import time


def make_image(image, flag):
    (h, w) = (image.shape[0], image.shape[1])

    mark_img, src_position = set_img_marker(image)
    wrap_img, minv = make_wrapping_img(image, src_position)
    # if flag:
    #     filter_img = make_filtering_img(wrap_img)
    # else:
    #     filter_img = tmp(wrap_img)
    filter_img = make_filtering_img(wrap_img)
    roi_img = set_roi_area(filter_img)

    # cv2.imshow("1111", mark_img)
    cv2.imshow("filter", roi_img)
    cv2.imshow("roi", set_roi_area(wrap_img))

    return minv, roi_img


def get_lane_information(original_image, roi_image, minv, correction):
    _DEG_ERROR_RANGE = 1
    _DIST_ERROR_RANGE = 3
    _DEVIATION_ERROR_RANGE = int(original_image.shape[1] / 10)

    left, right = find_lane(roi_image)
    # FIXME 이 경우에 왼쪽 또는 오른쪽으로 차를 틀어야 함
    if left == 0 or right == 0:
        now = time.localtime()
        # print(f"[%02d:%02d:%02d] 차선 미검출!" % (now.tm_hour, now.tm_min, now.tm_sec))
        if left == 0:
            raise Exception("LINE_L")
        elif right == 0:
            raise Exception("LINE_R")
        else:
            raise Exception("LINE")

    draw_info = get_lane_slope(roi_image, left, right)
    deg, dist, deviation, color_warp = draw_lane_lines(roi_image, minv, draw_info)
    result = add_img_weighted(original_image, color_warp, minv)

    # flag = True
    flag = dist > _DIST_ERROR_RANGE
    deg = deg if flag else 0

    direction = "LEFT" if ((deg < _DEG_ERROR_RANGE * -1) and flag) \
        else ("RIGHT" if ((deg > _DEG_ERROR_RANGE) and flag)
              else "FRONT")

    left = int(original_image.shape[1] * 0.03)
    top = int(original_image.shape[0] * 0.1)

    cv2.putText(result, f"[{direction}]", (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Deg : {deg}", (left, top + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Dist : {dist}", (left, top + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    deg = deg * 0.25 + correction

    # FIXME 중앙이 아닐 경우 차량 속도 감속 필요!!!
    # TODO deviation 이 음수이면 좌회전 / 양수이면 우회전 -> 각도 수정 필요
    deviation_result = "At Left" if deviation > _DEVIATION_ERROR_RANGE \
        else "At Right" if deviation < _DEVIATION_ERROR_RANGE * -1 \
        else "At Center"
    # TODO 차량이 차선 한쪽에 치우쳐 차량의 위치 보정이 필요한 경우
    # if direction == "FRONT" and deviation != 0:
    # deg = deg - 1 if deviation < 0 else deg + 1

    deviation_size = 0
    # if direction == "FRONT":
    if abs(deviation) > original_image.shape[1] / 3:
        deviation_size = 2.3
    elif abs(deviation) > original_image.shape[1] / 4:
        deviation_size = 1.5
    elif abs(deviation) > _DEVIATION_ERROR_RANGE:
        deviation_size = 1

    if deviation_result == "At Left":
        deg += deviation_size
    elif deviation_result == "At Right":
        deg -= deviation_size

    cv2.putText(result, f"[{deviation_result}] ", (left, top + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"Deg_M : {deg}", (left, top + 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return result, flag, deg


def main():
    # TODO 차량 연결 시, 활성화
    isTest = True

    speed = 0
    correction = 0
    laser, scan = 0, 0
    deg = 90

    if isTest:
        connection = make_serial_connection()
        laser, scan = make_connection()
    # MEMO 웹캠 정보 가져오기
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./video/result.avi")
    # cap = cv2.VideoCapture("./video/ex3.mp4")
    # cap = cv2.VideoCapture("./video/ex3_left-side.mp4")
    # cap = cv2.VideoCapture("./video/ex3_right-side.mp4")
    winname = "result"

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    file_name = time.strftime('[%Y-%m-%d]_%H-%M-%S', time.localtime(time.time())) + ".avi"

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(f"./test-log/{file_name}", fourcc, 30.0, (int(width), int(height)))

    cv2.namedWindow(winname)
    cv2.namedWindow("roi")
    cv2.namedWindow("filter")
    cv2.namedWindow("wrap")
    cv2.moveWindow(winname, 0, 0)
    cv2.moveWindow("roi", 850, 0)
    cv2.moveWindow("filter", 0, 550)
    cv2.moveWindow("wrap", 850, 550)

    start_flag = False
    sensing_flag = False
    filter_flag = True
    true_count = 0

    while True:
        # TODO 프레임 수정 필요
        ret, img = cap.read()
        key = cv2.waitKey(30)
        # img = cv2.imread("./img/cam_img.jpg")
        # MEMO 해상도에 따라 이미지 리사이징 필요
        # img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        if not ret:
            break

        if key == 32:
            start_flag = False if start_flag else True
        # if key == 32:  # CAR START or STOP
        #     speed = 9 if speed < 9 else 0
        # elif key == 119:  # SPEED UP
        #     if speed == 0:
        #         speed = 9
        #     else:
        #         speed += 1
        # elif key == 115:  # SPEED DOWN
        #     speed -= 1
        elif key == 97:
            correction -= 1
        elif key == 100:
            correction += 1

        if speed < 9:
            speed = 0

        result = sensing_lidar(laser, scan)
        
        if result:
            if true_count < 0:
                true_count = 0
            else:
                true_count += 1
        else:
            if true_count <= 0:
                true_count -= 1
            else:
                true_count = 0

        if true_count >= 5:
            sensing_flag = True
            # print("start")
        elif true_count <= -3:
            sensing_flag = False
            # print("stop")
        # else:
        #     print("wait") 

        # TODO REMOVE filter_flag
        filter_flag = False if filter_flag else True
        minv, roi_img = make_image(img, filter_flag)

        (h, w) = (img.shape[0], img.shape[1])
        cv2.line(img, (int(w / 2), 0), (int(w / 2), h), (255, 0, 0), 10)

        # if not start_flag:
        #     speed = 0

        # if not sensing_flag and speed > 0:
            # speed = 0

        car_flag = start_flag and sensing_flag
        # print(car_flag)

        if not car_flag:
            speed = 0

        try:
            result, is_curved, deg = get_lane_information(img, roi_img, minv, correction)
            
            # TODO IF is not curved car speed up
            if car_flag:
                speed = 9 if is_curved else 9
        except Exception as e:
            exp_msg = e.args[0]
            deg += correction
            result = img

            # print("------차량 속도를 낮춥니다------")
            if car_flag:
                speed = 9

            # MEMO 한 쪽 차선이 인식되지 않을 경우 예외처리
            if exp_msg == "LINE_L":
                print("------차를 왼쪽으로 틉니다------")
                deg -= 1.8

            elif exp_msg == "LINE_R":
                print("------차를 오른쪽으로 틉니다------")
                deg += 1.8

        order_msg = check_order(speed, deg)

        left = int(result.shape[1] * 0.03)
        top = int(result.shape[0] * 0.7)

        cv2.putText(result, f"{order_msg}", (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, f"Throttles : {speed}", (left, top + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, f"Deg_corr : {correction}", (left, top + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 차선 인식 값이 튀어, 갑자기 방향 전환되는 것을 방지
        # print(deg)
        if abs(deg) > 10:
            deg = 10 if deg > 0 else -10

        # print(speed)5

        if isTest:
            write_signal(connection, speed, deg)
        cv2.imshow(winname, result)
        out.write(result)

        if key & 0xFF == ord("q"):
            break

    laser.turnOff()
    laser.disconnecting()

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
