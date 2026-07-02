#include <math.h>

// ---------------- Motor Driver Pin ----------------
const int HANDLE_IN1 = 2;
const int HANDLE_IN2 = 3;

const int REAR_L_IN1 = 4;
const int REAR_L_IN2 = 5;
const int REAR_R_IN1 = 6;
const int REAR_R_IN2 = 7;

// ---------------- Steering Sensor ----------------
const int STEER_SENSOR_PIN = A0;

// A0 값이 512일 때 조향각 0도
const int STEER_CENTER_RAW = 512;

// 1 ADC count당 각도
// 네가 말한 조건: 1도는 270/1024 값
// 즉 각도 = ADC 변화량 * 270 / 1024
const float DEG_PER_ADC = 270.0f / 1024.0f;

// A0 값이 증가할 때 오른쪽(+)이면 1
// 반대로 움직이면 -1로 바꿔
const int STEER_SIGN = 1;

// 목표각 근처 허용 오차
const float STEER_TOLERANCE_DEG = 0.5f;

// 조향 모터 PWM
const int STEER_PWM = 150;

// ---------------- State Variables ----------------
float currentSteerDeg = 0.0f;
float targetSteerDeg = 0.0f;

int driveSpeed = 0;
char driveDir = 'S';

// 디버그 출력 간격
unsigned long lastDebugMs = 0;
const unsigned long DEBUG_INTERVAL_MS = 300;

void setup() {
  Serial.begin(9600);

  pinMode(HANDLE_IN1, OUTPUT);
  pinMode(HANDLE_IN2, OUTPUT);

  pinMode(REAR_L_IN1, OUTPUT);
  pinMode(REAR_L_IN2, OUTPUT);

  pinMode(REAR_R_IN1, OUTPUT);
  pinMode(REAR_R_IN2, OUTPUT);

  pinMode(STEER_SENSOR_PIN, INPUT);

  stopAllMotors();

  currentSteerDeg = readCurrentSteerDeg();
  targetSteerDeg = currentSteerDeg;

  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    Serial.print("RX: [");
    Serial.print(cmd);
    Serial.println("]");

    parseCommand(cmd);
  }

  applyDrive();
  applySteer();
}

// ---------------- 현재 조향각 읽기 ----------------

float readCurrentSteerDeg() {
  int raw = analogRead(STEER_SENSOR_PIN);

  float angle = (raw - STEER_CENTER_RAW) * DEG_PER_ADC * STEER_SIGN;

  return angle;
}

// ---------------- 핸들 제어 ----------------

void handleLeft() {
  analogWrite(HANDLE_IN1, STEER_PWM);
  analogWrite(HANDLE_IN2, 0);
}

void handleRight() {
  analogWrite(HANDLE_IN1, 0);
  analogWrite(HANDLE_IN2, STEER_PWM);
}

void handleStop() {
  analogWrite(HANDLE_IN1, 0);
  analogWrite(HANDLE_IN2, 0);
}

// ---------------- 뒷바퀴 제어 ----------------

void rearForward(int speed) {
  analogWrite(REAR_L_IN1, speed);
  analogWrite(REAR_L_IN2, 0);

  analogWrite(REAR_R_IN1, 0);
  analogWrite(REAR_R_IN2, speed);
}

void rearBackward(int speed) {
  analogWrite(REAR_L_IN1, 0);
  analogWrite(REAR_L_IN2, speed);

  analogWrite(REAR_R_IN1, speed);
  analogWrite(REAR_R_IN2, 0);
}

void rearStop() {
  analogWrite(REAR_L_IN1, 0);
  analogWrite(REAR_L_IN2, 0);

  analogWrite(REAR_R_IN1, 0);
  analogWrite(REAR_R_IN2, 0);
}

void stopAllMotors() {
  handleStop();
  rearStop();
}

// ---------------- 명령 파싱 ----------------

void parseCommand(String cmd) {
  if (cmd.length() == 0) return;

  if (cmd.startsWith("D,")) {
    parseDriveCommand(cmd);
  }
  else if (cmd.startsWith("S,")) {
    parseSteerCommand(cmd);
  }
  else {
    Serial.println("Invalid command");
  }
}

void parseDriveCommand(String cmd) {
  // 형식:
  // D,F,180
  // D,R,100
  // D,S,0

  int firstComma = cmd.indexOf(',');
  int secondComma = cmd.indexOf(',', firstComma + 1);

  if (firstComma == -1 || secondComma == -1) {
    Serial.println("Drive parse failed");
    return;
  }

  String dirStr = cmd.substring(firstComma + 1, secondComma);
  String speedStr = cmd.substring(secondComma + 1);

  char dir = dirStr.charAt(0);
  int speed = speedStr.toInt();

  speed = constrain(speed, 0, 255);

  if (dir == 'F' || dir == 'R' || dir == 'S') {
    driveDir = dir;
    driveSpeed = speed;

    Serial.print("Drive Dir: ");
    Serial.println(driveDir);

    Serial.print("Drive Speed: ");
    Serial.println(driveSpeed);
  }
  else {
    Serial.println("Invalid drive direction");
  }
}

void parseSteerCommand(String cmd) {
  // 형식:
  // S,-12.3
  // S,15.5
  // S,0.0

  int comma = cmd.indexOf(',');

  if (comma == -1) {
    Serial.println("Steer parse failed");
    return;
  }

  String angleStr = cmd.substring(comma + 1);
  float angle = angleStr.toFloat();

  // 소수점 첫째 자리로 반올림
  angle = round(angle * 10.0f) / 10.0f;

  targetSteerDeg = angle;

  Serial.print("Target Steer: ");
  Serial.println(targetSteerDeg, 1);
}

// ---------------- 실제 구동 적용 ----------------

void applyDrive() {
  if (driveDir == 'F') {
    rearForward(driveSpeed);
  }
  else if (driveDir == 'R') {
    rearBackward(driveSpeed);
  }
  else {
    rearStop();
  }
}

void applySteer() {
  currentSteerDeg = readCurrentSteerDeg();

  float steerError = targetSteerDeg - currentSteerDeg;

  if (fabs(steerError) <= STEER_TOLERANCE_DEG) {
    handleStop();
  }
  else if (steerError > 0) {
    handleRight();
  }
  else {
    handleLeft();
  }

  // 디버그 출력
  unsigned long now = millis();

  if (now - lastDebugMs >= DEBUG_INTERVAL_MS) {
    Serial.print("Current: ");
    Serial.print(currentSteerDeg, 1);

    Serial.print(" / Target: ");
    Serial.print(targetSteerDeg, 1);

    Serial.print(" / Error: ");
    Serial.println(steerError, 1);

    lastDebugMs = now;
  }
}