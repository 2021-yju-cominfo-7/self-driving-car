import serial
import cv2
import time


def write_signal(connection, throttles, direction, flag):
    order = check_order(throttles, direction, flag)
    now = time.localtime()

    print(f"[%02d:%02d:%02d] {order}" % (now.tm_hour, now.tm_min, now.tm_sec))
    connection.write(order)


def check_order(throttles, direction, flag):
    if flag:
        tmp = 7
    else:
        tmp = 9

    value_t = throttles
    value_d = direction + 90 - tmp

    if value_d <= 60:
        value_d = 60
    elif value_d >= 120:
        value_d = 120

    text_t = f"0{value_t}" if value_t < 10 else f"{value_t}"
    text_d = f"0{value_d}" if value_d < 100 else f"{value_d}"

    order = f"Q{text_t}{text_d}".encode("utf-8")
    return order


cap = cv2.VideoCapture(0)
PORT = "/dev/ttyACM0"
BaudRate = 9600
connection = serial.Serial(PORT, BaudRate, timeout=1)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./Arduino/result.avi', fourcc, 30.0, (int(width), int(height)))

speed = 0
correction = 0
deg = 0
correction_flag = True
while True:
    ret, img = cap.read()
    key = cv2.waitKey(1)

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

    correction_flag = False if correction_flag else True
    write_signal(connection, speed, deg + correction, correction_flag)
    cv2.imshow("main", img)
    out.write(img)

    if key & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
