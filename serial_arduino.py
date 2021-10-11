import serial


def make_serial_connection():
    PORT = "/dev/cu.usbmodem14401"
    BaudRate = 9600
    ser = serial.Serial(PORT, BaudRate, timeout=1)

    return ser


def write_signal(connection, throttles, direction):
    value_t = throttles
    value_d = direction + 90

    if value_d <= 60:
        value_d = 60
    elif value_d >= 120:
        value_d = 120

    text_t = f"0{value_t}" if value_t < 10 else f"{value_t}"
    text_d = f"0{value_d}" if value_d < 100 else f"{value_d}"

    order = f"Q{text_t}{text_d}".encode("utf-8")
    connection.write(order)


def check_order(throttles, direction):
    value_t = throttles
    value_d = direction + 90

    if value_d <= 60:
        value_d = 60
    elif value_d >= 120:
        value_d = 120

    text_t = f"0{value_t}" if value_t < 10 else f"{value_t}"
    text_d = f"0{value_d}" if value_d < 100 else f"{value_d}"

    order = f"Q{text_t}{text_d}".encode("utf-8")
    return order
