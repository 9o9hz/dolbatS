import serial
import time


# dol-soi.ino의 Serial.begin() 및 현재 Arduino Mega 연결 장치와 일치시킨다.
ser = serial.Serial("/dev/ttyACM0", 115200)
time.sleep(3)

MAX_STEER_DEG = 26.5

def send_drive(direction, speed):
    speed = max(0, min(255, speed))

    if direction not in ["F", "R", "S"]:
        raise ValueError("direction must be F, R, or S")

    cmd = f"D,{direction},{speed}\n"
    ser.write(cmd.encode())

def send_steer(angle):
    angle = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, angle))

    cmd = f"S,{angle}\n"
    ser.write(cmd.encode('ascii'))

TIME_DELTA = 0.07
ANGLE_DELTA = 4

for deg in range(0, -21, -ANGLE_DELTA):
    send_steer(deg)
    print(deg)
    time.sleep(TIME_DELTA)

while True:
    for deg in range(-20, 21, ANGLE_DELTA):
        send_steer(deg)
        print(deg)
        time.sleep(TIME_DELTA)

    for deg in range(21, -22, -ANGLE_DELTA):
        send_steer(deg)
        print(deg)
        time.sleep(TIME_DELTA)

# 시리얼 포트 닫기
ser.close()
