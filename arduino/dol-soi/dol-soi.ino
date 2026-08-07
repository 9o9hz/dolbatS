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
const char ULTRASONIC_SIDE_CODES[ULTRASONIC_SENSOR_COUNT] = {
  'L', 'L', 'R', 'R'
};
const char ULTRASONIC_POSITION_CODES[ULTRASONIC_SENSOR_COUNT] = {
  'F', 'R', 'F', 'R'
};

const unsigned long ULTRASONIC_TIMEOUT_US = 25000;
const int ULTRASONIC_FAILURE_THRESHOLD = 3;
// 네 센서를 하나씩 60 ms 간격으로 순차 측정한다.
// 같은 센서의 새 값은 약 240 ms(약 4.2 Hz)마다 생성된다.
const unsigned long ULTRASONIC_TRIGGER_INTERVAL_MS = 60;

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
int ultrasonicConsecutiveFailures[ULTRASONIC_SENSOR_COUNT] = {
  0, 0, 0, 0
};
bool ultrasonicTelemetryPending = false;
int pendingUltrasonicTelemetrySensor = 0;

// ---------------- Steering Sensor ----------------
const int STEER_SENSOR_PIN = A4;

// A4 값이 STEER_CENTER_RAW일 때 조향각 0도
const int STEER_CENTER_RAW = 560;
// 센서/링크를 정비한 뒤에는 실차 중앙의 raw 값으로 반드시 재보정한다.

// 1 ADC count당 각도
// 네가 말한 조건: 1도는 270/1024 값
// 즉 각도 = ADC 변화량 * 270 / 1024
const float DEG_PER_ADC = 53.0f / 266.0f;

// 조향각 규약: 왼쪽은 양수(+), 오른쪽은 음수(-).
// 실차에서는 왼쪽으로 움직일 때 A4 값이 증가한다. 이 극성이 틀리면
// 위치 오차를 줄이지 않고 같은 방향으로 계속 구동해 기계적 끝단에 닿는다.
const int STEER_SIGN = 1;
const float MAX_STEER_DEG = 26.5f;

// 조향 센서의 안전 동작 범위와 목표값 허용 오차 (ADC raw)
// 현재 센서는 왼쪽으로 갈수록 raw가 커지고 오른쪽으로 갈수록 작아짐
const int STEER_RAW_LIMIT_OFFSET =
  (int)(MAX_STEER_DEG / DEG_PER_ADC + 0.5f);
const int STEER_RAW_MIN = STEER_CENTER_RAW - STEER_RAW_LIMIT_OFFSET;
const int STEER_RAW_MAX = STEER_CENTER_RAW + STEER_RAW_LIMIT_OFFSET;
const int STEER_RAW_TOLERANCE = 2;

// 센서 단선/링크 이탈 시 조향 모터를 끝단까지 계속 구동하지 않는다.
const unsigned long STEER_PROGRESS_TIMEOUT_MS = 250;
const int STEER_MIN_PROGRESS_RAW = 2;

// 조향 모터 PWM
const int STEER_PWM = 160;

// ---------------- State Variables ----------------
float currentSteerDeg = 0.0f;
int currentSteerRaw = STEER_CENTER_RAW;
int targetSteerRaw = STEER_CENTER_RAW;
int steerProgressDirection = 0;
int steerProgressStartRaw = STEER_CENTER_RAW;
unsigned long steerProgressStartMs = 0;
bool steerFeedbackFault = false;

int driveSpeed = 0;
char driveDir = 'S';

// PC/ROS/USB 통신이 끊기면 마지막 구동 명령을 무기한 유지하지 않는다.
// unsigned long 뺄셈은 millis() overflow에도 안전하다.
const unsigned long DRIVE_COMMAND_TIMEOUT_MS = 500;
unsigned long lastValidDriveCommandMs = 0;
bool hasReceivedDriveCommand = false;
bool driveWatchdogStopped = false;

// 줄바꿈으로 끝나는 명령을 기다리는 동안 제어 루프가 멈추지 않도록
// 도착한 바이트만 고정 크기 버퍼에 저장한다.
const uint8_t SERIAL_BUFFER_SIZE = 40;
char serialBuffer[SERIAL_BUFFER_SIZE];
uint8_t serialBufferLength = 0;
bool discardSerialUntilNewline = false;

// 차량 상태는 초음파 측정 주기와 독립적으로 100 Hz로 송출한다.
const unsigned long VEHICLE_TELEMETRY_INTERVAL_MS = 10;
unsigned long lastVehicleTelemetryMs = 0;

void setup() {
  Serial.begin(115200);

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
  updateSerialCommands();

  updateDriveWatchdog();
  applyDrive();
  applySteer();
  updateUltrasonicSensors();
  publishVehicleTelemetry();
  publishUltrasonicTelemetry();
}

// ---------------- 비차단 명령 수신 ----------------

void updateSerialCommands() {
  while (Serial.available() > 0) {
    char received = static_cast<char>(Serial.read());

    if (received == '\r') {
      continue;
    }

    if (received == '\n') {
      if (!discardSerialUntilNewline && serialBufferLength > 0) {
        serialBuffer[serialBufferLength] = '\0';
        parseCommand(String(serialBuffer));
      }

      serialBufferLength = 0;
      discardSerialUntilNewline = false;
      continue;
    }

    if (discardSerialUntilNewline) {
      continue;
    }

    if (serialBufferLength < SERIAL_BUFFER_SIZE - 1) {
      serialBuffer[serialBufferLength++] = received;
    }
    else {
      // 너무 긴 명령의 뒷부분을 정상 명령으로 잘못 해석하지 않는다.
      serialBufferLength = 0;
      discardSerialUntilNewline = true;
    }
  }
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

float filterUltrasonicDistance(int sensor, float newDistance) {
  float previous = ultrasonicDistanceCm[sensor];

  if (previous < 0.0f) {
    return newDistance;
  }

  // 가까운 장애물은 지연 없이 반영한다.
  if (newDistance < previous) {
    return newDistance;
  }

  // 장애물이 사라졌다는 판단은 완만하게 반영한다.
  const float alpha = 0.25f;
  return previous + alpha * (newDistance - previous);
}

void finishUltrasonicMeasurement(float distanceCm) {
  if (distanceCm >= 0.0f) {
    ultrasonicDistanceCm[activeUltrasonicSensor] =
      filterUltrasonicDistance(activeUltrasonicSensor, distanceCm);
    ultrasonicConsecutiveFailures[activeUltrasonicSensor] = 0;
  }
  else {
    int &failureCount =
      ultrasonicConsecutiveFailures[activeUltrasonicSensor];
    if (failureCount < ULTRASONIC_FAILURE_THRESHOLD) {
      failureCount++;
    }
    if (failureCount >= ULTRASONIC_FAILURE_THRESHOLD) {
      ultrasonicDistanceCm[activeUltrasonicSensor] = -1.0f;
    }
  }

  ultrasonicState = ULTRASONIC_IDLE;
  lastUltrasonicTriggerMs = millis();
  nextUltrasonicSensor =
    (activeUltrasonicSensor + 1) % ULTRASONIC_SENSOR_COUNT;
  pendingUltrasonicTelemetrySensor = activeUltrasonicSensor;
  ultrasonicTelemetryPending = true;
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

void publishVehicleTelemetry() {
  unsigned long now = millis();
  if (
    now - lastVehicleTelemetryMs
      < VEHICLE_TELEMETRY_INTERVAL_MS
  ) {
    return;
  }

  int signedSpeed = 0;
  if (driveDir == 'F') {
    signedSpeed = driveSpeed;
  }
  else if (driveDir == 'R') {
    signedSpeed = -driveSpeed;
  }

  // 형식: VEH,signed_drive_pwm,current_steer
  // 속도 필드는 측정 속도가 아니라 명령된 PWM(-255~255)이다.
  Serial.print("VEH,");
  Serial.print(signedSpeed);
  Serial.print(",");
  Serial.println(currentSteerDeg, 1);

  lastVehicleTelemetryMs = now;
}

void publishUltrasonicTelemetry() {
  if (!ultrasonicTelemetryPending) {
    return;
  }

  // 새 측정이 끝난 센서의 값만 송출한다.
  // 형식: ULT,side(L/R),position(F/R),distance_cm
  int sensor = pendingUltrasonicTelemetrySensor;
  Serial.print("ULT,");
  Serial.print(ULTRASONIC_SIDE_CODES[sensor]);
  Serial.print(",");
  Serial.print(ULTRASONIC_POSITION_CODES[sensor]);
  Serial.print(",");
  Serial.println(ultrasonicDistanceCm[sensor], 1);

  ultrasonicTelemetryPending = false;
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
  // 실차에서 A4 raw가 증가하는 방향
  analogWrite(HANDLE_IN1, 0);
  analogWrite(HANDLE_IN2, STEER_PWM);
}

void handleRight() {
  // 실차에서 A4 raw가 감소하는 방향
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
  int newTargetSteerRaw = constrain(
    requestedRaw,
    STEER_RAW_MIN,
    STEER_RAW_MAX
  );
  targetSteerRaw = newTargetSteerRaw;
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
    steerFeedbackFault = false;
    steerProgressDirection = 0;
  }
  else if (steerFeedbackFault) {
    handleStop();
  }
  else {
    int requiredRawDirection = steerRawError > 0 ? 1 : -1;
    unsigned long nowMs = millis();
    if (requiredRawDirection != steerProgressDirection) {
      steerProgressDirection = requiredRawDirection;
      steerProgressStartRaw = currentSteerRaw;
      steerProgressStartMs = nowMs;
    }
    else if (
      (unsigned long)(nowMs - steerProgressStartMs)
        >= STEER_PROGRESS_TIMEOUT_MS
    ) {
      int progressRaw = requiredRawDirection
        * (currentSteerRaw - steerProgressStartRaw);
      if (progressRaw < STEER_MIN_PROGRESS_RAW) {
        steerFeedbackFault = true;
        handleStop();
        return;
      }
      steerProgressStartRaw = currentSteerRaw;
      steerProgressStartMs = nowMs;
    }

    if (requiredRawDirection > 0 && currentSteerRaw < STEER_RAW_MAX) {
      handleLeft();
    }
    else if (requiredRawDirection < 0 && currentSteerRaw > STEER_RAW_MIN) {
      handleRight();
    }
    else {
      handleStop();
    }
  }
}
