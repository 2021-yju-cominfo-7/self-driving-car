import serial
import time

PORT = "/dev/cu.usbmodem144101"
BaudRate = 9600
ser = serial.Serial(PORT, BaudRate, timeout=1)

while True:
    if ser.readable():
        val = input("입력 : ")

        if val == '1':
            val = val.encode('utf-8')
            ser.write(val)
            print("LED TURNED ON")
            time.sleep(0.5)

        elif val == '0':
            val = val.encode('utf-8')
            ser.write(val)
            print("LED TURNED OFF")
            time.sleep(0.5)
