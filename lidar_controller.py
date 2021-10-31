import ydlidar
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi

RMAX = 32.0

lidar_polar = plt.subplot(polar=True)
lidar_polar.autoscale_view(True,True,True)
lidar_polar.set_rmax(RMAX)
lidar_polar.grid(True)
ports = ydlidar.lidarPortList()
port = "/dev/ydlidar"
for key, value in ports.items():
    port = value
    
laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 9)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)

scan = ydlidar.LaserScan()
toDegree = 180 / pi

def sensing_lidar():
    lidar_flag = True
    r = laser.doProcessSimple(scan)
    if r:
        angle = []
        ran = []
        intensity = []
        for point in scan.points:
            angle.append(point.angle * -1)

            tmpAngle = point.angle * -1 * toDegree
            range = point.range * 100

            if abs(tmpAngle) >= 135:
                ran.append(range)

                if 1 < range <= 30:
                    # FRONT SIDE
                    if abs(tmpAngle) >= 160:
                        # print("FRONT STOP")
                        lidar_flag = False
                    # RIGHT SIDE
                    elif point.angle * -1 * toDegree >= 135:
                        # print("RIGHT STOP")
                        lidar_flag = False
                    # LEFT SIDE
                    elif point.angle * -1 * toDegree <= -135:
                        # print("LEFT STOP")
                        lidar_flag = False
            else:
                ran.append(0)
            intensity.append(point.intensity)
        
        lidar_polar.clear()
        lidar_polar.scatter(angle, ran, c=intensity, cmap='hsv', alpha=1)

    return lidar_flag

ret = laser.initialize()
if ret:
    ret = laser.turnOn()
    true_count = 0
    
    while True:
        result = sensing_lidar()

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

        if true_count >= 10:
            print("start")
        elif true_count <= -3:
            print("stop")
        else:
            print("wait")

    laser.turnOff()

laser.disconnecting()
plt.close()


