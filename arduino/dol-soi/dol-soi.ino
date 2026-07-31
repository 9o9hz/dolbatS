#include <math.h>

// ---------------- Motor Driver Pin ----------------
const int HANDLE_IN1 = 2;
const int HANDLE_IN2 = 3;

const int REAR_L_IN1 = 4;
const int REAR_L_IN2 = 5;
const int REAR_R_IN1 = 6;
const int REAR_R_IN2 = 7;

// ---------------- Ultrasonic Sensors ----------------
const int LEFT_FRONT_ULTRASONIC_ECHO = 22;
const int LEFT_FRONT_ULTRASONIC_TRIG = 23;
const int LEFT_REAR_ULTRASONIC_ECHO = 24;
const int LEFT_REAR_ULTRASONIC_TRIG = 25;
const int RIGHT_FRONT_ULTRASONIC_ECHO = 26;
const int RIGHT_FRONT_ULTRASONIC_TRIG = 27;
const int RIGHT_REAR_ULTRASONIC_ECHO = 28;
const int RIGHT_REAR_ULTRASONIC_TRIG = 29;

const int ULTRASONIC_SENSOR_COUNT = 4;
const int ULTRASONIC_ECHO_PINS[ULTRASONIC_SENSOR_COUNT] = {
  LEFT_FRONT_ULTRASONIC_ECHO,
  LEFT_REAR_ULTRASONIC_ECHO,
  RIGHT_FRONT_ULTRASONIC_ECHO,
  RIGHT_REAR_ULTRASONIC_ECHO
};
const int ULTRASONIC_TRIG_PINS[ULTRASONIC_SENSOR_COUNT] = {
  LEFT_FRONT_ULTRASONIC_TRIG,
  LEFT_REAR_ULTRASONIC_TRIG,
  RIGHT_FRONT_ULTRASONIC_TRIG,
  RIGHT_REAR_ULTRASONIC_TRIG
};

const unsigned long ULTRASONIC_TIMEOUT_US = 25000;
const unsigned long ULTRASONIC_TRIGGER_INTERVAL_MS = 30;

enum UltrasonicState {
  ULTRASONIC_IDLE,
  ULTRASONIC_WAIT_RISE,
  ULTRASONIC_WAIT_FALL
};

UltrasonicState ultrasonicState = ULTRASONIC_IDLE;
unsigned long ultrasonicTriggerStartUs = 0;
unsigned long ultrasonicEchoStartUs = 0;
unsigned long lastUltrasonicTriggerMs = 0;
int activeUltrasonicSensor = 0;
int nextUltrasonicSensor = 0;
float ultrasonicDistanceCm[ULTRASONIC_SENSOR_COUNT] = {
  -1.0f, -1.0f, -1.0f, -1.0f
};

// ---------------- Steering Sensor ----------------
const int STEER_SENSOR_PIN = A4;

// A4 값이 STEER_CENTER_RAW일 때 조향각 0도
const int STEER_CENTER_RAW = 446;

// 1 ADC count당 각도
// 네가 말한 조건: 1도는 270/1024 값
// 즉 각도 = ADC 변화량 * 270 / 1024
const float DEG_PER_ADC = 270.0f / 1024.0f;

// 조향각 규약: 왼쪽은 양수(+), 오른쪽은 음수(-)
// 현재 센서는 오른쪽으로 움직일 때 A4 값이 증가하므로 -1
const int STEER_SIGN = -1;
const float MAX_STEER_DEG = 25.0f;

// 조향 센서의 안전 동작 범위와 목표값 허용 오차 (ADC raw)
// 현재 센서는 왼쪽으로 갈수록 raw가 작아지고 오른쪽으로 갈수록 커짐
const int STEER_RAW_LIMIT_OFFSET =
  (int)(MAX_STEER_DEG / DEG_PER_ADC + 0.5f);
const int STEER_RAW_MIN = STEER_CENTER_RAW - STEER_RAW_LIMIT_OFFSET;
const int STEER_RAW_MAX = STEER_CENTER_RAW + STEER_RAW_LIMIT_OFFSET;
const int STEER_RAW_TOLERANCE = 2;

// 조향 모터 PWM
const int STEER_PWM = 150;

// ---------------- State Variables ----------------
float currentSteerDeg = 0.0f;
int currentSteerRaw = STEER_CENTER_RAW;
int targetSteerRaw = STEER_CENTER_RAW;

int driveSpeed = 0;
char driveDir = 'S';

// PC/ROS/USB 통신이 끊기면 마지막 구동 명령을 무기한 유지하지 않는다.
// unsigned long 뺄셈은 millis() overflow에도 안전하다.
const unsigned long DRIVE_COMMAND_TIMEOUT_MS = 500;
unsigned long lastValidDriveCommandMs = 0;
bool hasReceivedDriveCommand = false;
bool driveWatchdogStopped = false;

const unsigned long TELEMETRY_INTERVAL_MS = 10;
unsigned long lastTelemetryMs = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5);

  pinMode(HANDLE_IN1, OUTPUT);
  pinMode(HANDLE_IN2, OUTPUT);

  pinMode(REAR_L_IN1, OUTPUT);
  pinMode(REAR_L_IN2, OUTPUT);

  pinMode(REAR_R_IN1, OUTPUT);
  pinMode(REAR_R_IN2, OUTPUT);

  for (int sensor = 0; sensor < ULTRASONIC_SENSOR_COUNT; sensor++) {
    pinMode(ULTRASONIC_TRIG_PINS[sensor], OUTPUT);
    pinMode(ULTRASONIC_ECHO_PINS[sensor], INPUT);
    digitalWrite(ULTRASONIC_TRIG_PINS[sensor], LOW);
  }

  pinMode(STEER_SENSOR_PIN, INPUT);

  stopAllMotors();

  currentSteerRaw = readSteerSensorRaw();
  currentSteerDeg = steerRawToDeg(currentSteerRaw);
  targetSteerRaw = constrain(
    currentSteerRaw,
    STEER_RAW_MIN,
    STEER_RAW_MAX
  );
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    parseCommand(cmd);
  }

  updateDriveWatchdog();
  applyDrive();
  applySteer();
  updateUltrasonicSensors();
  publishTelemetry();
}

// ---------------- 비차단 초음파 거리 측정 ----------------

void startUltrasonicMeasurement(int sensor) {
  int trigPin = ULTRASONIC_TRIG_PINS[sensor];

  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  activeUltrasonicSensor = sensor;
  ultrasonicTriggerStartUs = micros();
  ultrasonicState = ULTRASONIC_WAIT_RISE;
}

void finishUltrasonicMeasurement(float distanceCm) {
  ultrasonicDistanceCm[activeUltrasonicSensor] = distanceCm;

  ultrasonicState = ULTRASONIC_IDLE;
  lastUltrasonicTriggerMs = millis();
  nextUltrasonicSensor =
    (activeUltrasonicSensor + 1) % ULTRASONIC_SENSOR_COUNT;
}

void updateUltrasonicSensors() {
  unsigned long nowUs = micros();
  int activeEchoPin = ULTRASONIC_ECHO_PINS[activeUltrasonicSensor];

  if (ultrasonicState == ULTRASONIC_WAIT_RISE) {
    if (digitalRead(activeEchoPin) == HIGH) {
      ultrasonicEchoStartUs = nowUs;
      ultrasonicState = ULTRASONIC_WAIT_FALL;
    }
    else if (nowUs - ultrasonicTriggerStartUs >= ULTRASONIC_TIMEOUT_US) {
      finishUltrasonicMeasurement(-1.0f);
    }
    return;
  }

  if (ultrasonicState == ULTRASONIC_WAIT_FALL) {
    if (digitalRead(activeEchoPin) == LOW) {
      unsigned long durationUs = nowUs - ultrasonicEchoStartUs;
      float distanceCm = durationUs * 0.0343f / 2.0f;
      finishUltrasonicMeasurement(distanceCm <= 400.0f ? distanceCm : -1.0f);
    }
    else if (nowUs - ultrasonicEchoStartUs >= ULTRASONIC_TIMEOUT_US) {
      finishUltrasonicMeasurement(-1.0f);
    }
    return;
  }

  if (millis() - lastUltrasonicTriggerMs < ULTRASONIC_TRIGGER_INTERVAL_MS) {
    return;
  }

  // 한 번에 하나만 발사해 네 센서 사이의 초음파 간섭을 줄인다.
  startUltrasonicMeasurement(nextUltrasonicSensor);
}

// ---------------- 상태 송출 ----------------

void publishTelemetry() {
  unsigned long now = millis();
  if (now - lastTelemetryMs < TELEMETRY_INTERVAL_MS) {
    return;
  }

  int signedSpeed = 0;
  if (driveDir == 'F') {
    signedSpeed = driveSpeed;
  }
  else if (driveDir == 'R') {
    signedSpeed = -driveSpeed;
  }

  // 형식: signed_drive_pwm,current_steer,left_front_cm,left_rear_cm,
  //       right_front_cm,right_rear_cm
  // 첫 필드는 측정 속도가 아니라 명령된 PWM(-255~255)이다.
  Serial.print(signedSpeed);
  Serial.print(",");
  Serial.print(currentSteerDeg, 1);
  Serial.print(",");
  Serial.print(ultrasonicDistanceCm[0], 1);
  Serial.print(",");
  Serial.print(ultrasonicDistanceCm[1], 1);
  Serial.print(",");
  Serial.print(ultrasonicDistanceCm[2], 1);
  Serial.print(",");
  Serial.println(ultrasonicDistanceCm[3], 1);

  lastTelemetryMs = now;
}

// ---------------- 현재 조향각 읽기 ----------------

int readSteerSensorRaw() {
  return analogRead(STEER_SENSOR_PIN);
}

float steerRawToDeg(int raw) {
  float angle = (raw - STEER_CENTER_RAW) * DEG_PER_ADC * STEER_SIGN;

  return angle;
}

// ---------------- 핸들 제어 ----------------

void handleLeft() {
  // A4 raw가 감소하는 방향
  analogWrite(HANDLE_IN1, 0);
  analogWrite(HANDLE_IN2, STEER_PWM);
}

void handleRight() {
  // A4 raw가 증가하는 방향
  analogWrite(HANDLE_IN1, STEER_PWM);
  analogWrite(HANDLE_IN2, 0);
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
}

void parseDriveCommand(String cmd) {
  // 형식:
  // D,F,180
  // D,R,100
  // D,S,0

  int firstComma = cmd.indexOf(',');
  int secondComma = cmd.indexOf(',', firstComma + 1);

  if (firstComma == -1 || secondComma == -1) {
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
    lastValidDriveCommandMs = millis();
    hasReceivedDriveCommand = true;
    driveWatchdogStopped = false;
  }
}

void updateDriveWatchdog() {
  if (
    hasReceivedDriveCommand
    && !driveWatchdogStopped
    && (unsigned long)(millis() - lastValidDriveCommandMs)
      > DRIVE_COMMAND_TIMEOUT_MS
  ) {
    driveDir = 'S';
    driveSpeed = 0;
    rearStop();
    driveWatchdogStopped = true;
  }
}

void parseSteerCommand(String cmd) {
  // 형식:
  // S,-12.3
  // S,15.5
  // S,0.0

  int comma = cmd.indexOf(',');

  if (comma == -1) {
    return;
  }

  String angleStr = cmd.substring(comma + 1);
  float angle = angleStr.toFloat();

  // 소수점 첫째 자리로 반올림
  angle = round(angle * 10.0f) / 10.0f;
  angle = constrain(angle, -MAX_STEER_DEG, MAX_STEER_DEG);

  // 각도 명령을 센서 raw 목표값으로 변환한 뒤 안전 범위로 제한
  int requestedRaw = round(
    STEER_CENTER_RAW + angle / (DEG_PER_ADC * STEER_SIGN)
  );
  targetSteerRaw = constrain(
    requestedRaw,
    STEER_RAW_MIN,
    STEER_RAW_MAX
  );
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
  currentSteerRaw = readSteerSensorRaw();
  currentSteerDeg = steerRawToDeg(currentSteerRaw);

  int steerRawError = targetSteerRaw - currentSteerRaw;

  if (abs(steerRawError) <= STEER_RAW_TOLERANCE) {
    handleStop();
  }
  else if (steerRawError < 0 && currentSteerRaw > STEER_RAW_MIN) {
    handleLeft();
  }
  else if (steerRawError > 0 && currentSteerRaw < STEER_RAW_MAX) {
    handleRight();
  }
  else {
    handleStop();
  }
}
