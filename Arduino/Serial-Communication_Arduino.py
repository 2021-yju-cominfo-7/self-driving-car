import serial
import time
import cv2

winname = "Controller"
throttles = "Throttles"
direction = "Direction"

image = cv2.imread("../img/slope_test.jpg")
cv2.namedWindow(winname)

cv2.createTrackbar(throttles, winname, 0, 20, lambda x: x)
cv2.setTrackbarPos(throttles, winname, 0)

cv2.createTrackbar(direction, winname, 0, 60, lambda x: x)
cv2.setTrackbarPos(direction, winname, 30)

# PORT = "/dev/cu.usbmodem144101"
PORT = "/dev/cu.usbmodem14401"
# PORT = "/dev/cu.RNBT-8138-RNI-SPP"
BaudRate = 9600
ser = serial.Serial(PORT, BaudRate, timeout=1)

while True:
    value_t = cv2.getTrackbarPos(throttles, winname)
    value_d = cv2.getTrackbarPos(direction, winname) + 60
    text_t = f"0{value_t}" if value_t < 10 else f"{value_t}"
    text_d = f"0{value_d}" if value_d < 100 else f"{value_d}"

    result = f"Q{text_t}{text_d}".encode("utf-8")
    ser.write(result)

    print(f"{result} // {throttles} : {text_t} / {direction} : {text_d}")

    cv2.imshow(winname, image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
