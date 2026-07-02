import keyboard
import serial
import time

print("방향키를 눌러보세요. (종료하려면 'q'를 누르세요)")

# 현재 어떤 명령이 전송되었는지 기록하는 변수 (중복 전송 방지)
# '0': 정지, '1': 좌, '2': 우, '3': 상, '4': 하
current_state = '0'

deg = 0

# 시리얼 포트 설정 (사용자 환경에 맞게 변경)
ser = serial.Serial("COM9", 9600)

def send_drive(direction, speed):
    speed = max(0, min(255, speed))

    if direction not in ["F", "R", "S"]:
        raise ValueError("direction must be F, R, or S")

    cmd = f"D,{direction},{speed}\n"
    ser.write(cmd.encode('ascii'))

def send_steer(angle):
    angle = max(-21, min(21, angle))

    cmd = f"S,{angle}\n"
    ser.write(cmd.encode('ascii'))

while True:
    # 1. 왼쪽 방향키 감지
    if keyboard.is_pressed('left'):
        if current_state != '1':  # 이전에 '1'을 보낸 게 아니라면 (처음 눌렸다면)
            print("왼쪽 방향키 감지")
            deg -= 1
            send_steer(deg)
            current_state = '1'

    # 2. 오른쪽 방향키 감지
    elif keyboard.is_pressed('right'):
        if current_state != '2':
            print("오른쪽 방향키 감지")
            deg += 1
            send_steer(deg)
            current_state = '2'

    # 3. 위쪽 방향키 감지
    elif keyboard.is_pressed('up'):
        if current_state != '3':
            print("위쪽 방향키 감지")
            send_drive("F", 255)
            current_state = '3'

    # 4. 아래쪽 방향키 감지
    elif keyboard.is_pressed('down'):
        if current_state != '4':
            print("아래쪽 방향키 감지")
            send_drive("R", 255)
            current_state = '4'

        # 4. 아래쪽 방향키 감지
    elif keyboard.is_pressed('a'):
        if current_state != '5':
            input()
            deg = int(input("몇 도로 갈 거에요? (최대 +-21도) : "))
            print(deg, "도로 회전")
            send_steer(deg)
            current_state = '5'

    # 5. 종료 키 감지
    elif keyboard.is_pressed('q'):
        print("프로그램을 종료합니다.")
        # 종료 전 정지 신호 송신
        send_drive('S', 0)
        break

    else:
        send_drive('S', 0)


    # CPU 점유율이 100%까지 치솟는 것을 방지하기 위한 미세한 대기 시간
    time.sleep(0.01)

# 시리얼 포트 닫기
ser.close()

