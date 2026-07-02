import keyboard
import serial
import time

print("방향키를 눌러보세요. 종료하려면 q를 누르세요.")

ser = serial.Serial("COM3", 9600, timeout=0.1)
time.sleep(2)  # 아두이노 리셋 대기

deg = 0

last_drive = None
last_steer = None
last_left_time = 0
last_right_time = 0

STEER_INTERVAL = 0.08  # 좌우 키를 누르고 있을 때 각도 변경 간격


def send_drive(direction, speed):
    global last_drive

    speed = max(0, min(255, speed))

    if direction not in ["F", "R", "S"]:
        raise ValueError("direction must be F, R, or S")

    cmd = f"D,{direction},{speed}\n"

    # 같은 주행 명령 중복 전송 방지
    if last_drive != cmd:
        ser.write(cmd.encode("utf-8"))
        print("TX:", cmd.strip())
        last_drive = cmd


def send_steer(angle):
    global last_steer

    angle = max(-21, min(21, angle))

    cmd = f"S,{angle}\n"

    # 같은 조향 명령 중복 전송 방지
    if last_steer != cmd:
        ser.write(cmd.encode("utf-8"))
        print("TX:", cmd.strip())
        last_steer = cmd


try:
    while True:
        now = time.time()

        # 종료
        if keyboard.is_pressed("q"):
            print("프로그램을 종료합니다.")
            send_drive("S", 0)
            send_steer(0)
            break

        # 조향 왼쪽
        if keyboard.is_pressed("left"):
            if now - last_left_time > STEER_INTERVAL:
                deg -= 1
                deg = max(-21, min(21, deg))
                print("왼쪽 조향:", deg)
                send_steer(deg)
                last_left_time = now

        # 조향 오른쪽
        elif keyboard.is_pressed("right"):
            if now - last_right_time > STEER_INTERVAL:
                deg += 1
                deg = max(-21, min(21, deg))
                print("오른쪽 조향:", deg)
                send_steer(deg)
                last_right_time = now

        # 전진
        if keyboard.is_pressed("up"):
            send_drive("F", 255)

        # 후진
        elif keyboard.is_pressed("down"):
            send_drive("R", 255)

        # 위/아래 안 누르면 정지
        else:
            send_drive("S", 0)

        # a 키로 각도 직접 입력
        if keyboard.is_pressed("a"):
            time.sleep(0.2)
            try:
                deg = int(input("몇 도로 갈 거에요? 최대 ±21도: "))
                deg = max(-21, min(21, deg))
                send_steer(deg)
            except ValueError:
                print("숫자를 입력하세요.")

        # 아두이노 응답 출력
        while ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print("ARDUINO:", line)

        time.sleep(0.01)

finally:
    send_drive("S", 0)
    send_steer(0)
    ser.close()