import serial
import time


# 시리얼 포트 설정 (사용자 환경에 맞게 변경)
ser = serial.Serial("COM9", 9600)
time.sleep(3)

def send_drive(direction, speed):
    speed = max(0, min(255, speed))

    if direction not in ["F", "R", "S"]:
        raise ValueError("direction must be F, R, or S")

    cmd = f"D,{direction},{speed}\n"
    ser.write(cmd.encode())

def send_steer(angle):
    angle = max(-21, min(21, angle))

    cmd = f"S,{angle}\n"
    ser.write(cmd.encode('ascii'))

TIME_DELTA = 0.07
ANGLE_DELTA = 4

for deg in range(0, -22, -ANGLE_DELTA):
    send_steer(deg)
    print(deg)
    time.sleep(TIME_DELTA)

while True:
    for deg in range(-21, 22, ANGLE_DELTA):
        send_steer(deg)
        print(deg)
        time.sleep(TIME_DELTA)

    for deg in range(21, -22, -ANGLE_DELTA):
        send_steer(deg)
        print(deg)
        time.sleep(TIME_DELTA)

# 시리얼 포트 닫기
ser.close()

