import serial
import time


def make_serial_connection():
    PORT = "/dev/cu.usbmodem14301"
    # PORT = "/dev/ttyACM0"
    BaudRate = 9600
    ser = serial.Serial(PORT, BaudRate, timeout=1)

    return ser


def write_signal(connection, throttles, direction):
    order = check_order(throttles, direction)
    now = time.localtime()

    # print(f"[%02d:%02d:%02d] {order}" % (now.tm_hour, now.tm_min, now.tm_sec))
    connection.write(order)


def check_order(throttles, direction):
    value_t = throttles
    value_d = direction + 90 - 7

    if value_d <= 60:
        value_d = 60
    elif value_d >= 120:
        value_d = 120

    text_t = f"0{value_t}" if value_t < 10 else f"{value_t}"
    text_d = f"0{value_d}" if value_d < 100 else f"{value_d}"

    order = f"Q{text_t}{text_d}".encode("utf-8")
    return order
