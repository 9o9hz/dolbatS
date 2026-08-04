/*
 * Manual steering motor + potentiometer monitor
 *
 * Serial commands:
 *   L : rotate steering motor continuously to the left
 *   R : rotate steering motor continuously to the right
 *   S : stop steering motor
 *
 * Potentiometer value is printed every 50 ms.
 *
 * Serial monitor:
 *   Baud rate: 115200
 *
 * WARNING:
 *   This sketch does not use automatic steering limits.
 *   Press S before reaching a mechanical end stop.
 */

const uint8_t STEER_MOTOR_IN1 = 2;
const uint8_t STEER_MOTOR_IN2 = 3;
const uint8_t STEER_POT_PIN   = A4;

// Motor PWM: 0~255
const uint8_t STEER_PWM = 150;

// Potentiometer output interval
const unsigned long PRINT_INTERVAL_MS = 50;

enum MotorState {
  MOTOR_STOPPED,
  MOTOR_LEFT,
  MOTOR_RIGHT
};

MotorState motorState = MOTOR_STOPPED;
unsigned long lastPrintMs = 0;

const char* motorStateName() {
  switch (motorState) {
    case MOTOR_LEFT:
      return "LEFT";

    case MOTOR_RIGHT:
      return "RIGHT";

    default:
      return "STOP";
  }
}

void motorStop() {
  analogWrite(STEER_MOTOR_IN1, 0);
  analogWrite(STEER_MOTOR_IN2, 0);
  motorState = MOTOR_STOPPED;
  Serial.println(F("COMMAND,STOP"));
}

void motorLeft() {
  // Existing steering convention:
  // left = IN1 off, IN2 PWM
  analogWrite(STEER_MOTOR_IN1, 0);
  analogWrite(STEER_MOTOR_IN2, STEER_PWM);
  motorState = MOTOR_LEFT;

  Serial.print(F("COMMAND,LEFT,PWM="));
  Serial.println(STEER_PWM);
}

void motorRight() {
  // Existing steering convention:
  // right = IN1 PWM, IN2 off
  analogWrite(STEER_MOTOR_IN1, STEER_PWM);
  analogWrite(STEER_MOTOR_IN2, 0);
  motorState = MOTOR_RIGHT;

  Serial.print(F("COMMAND,RIGHT,PWM="));
  Serial.println(STEER_PWM);
}

void printHelp() {
  Serial.println();
  Serial.println(F("=== Manual steering motor test ==="));
  Serial.println(F("L : continuous LEFT"));
  Serial.println(F("R : continuous RIGHT"));
  Serial.println(F("S : STOP"));
  Serial.print(F("PWM="));
  Serial.println(STEER_PWM);
  Serial.println(F("OUTPUT: POT,time_ms,raw,state"));
  Serial.println(F("WARNING: No automatic potentiometer limit protection."));
  Serial.println();
}

void processCommand(char command) {
  switch (command) {
    case 'L':
    case 'l':
      motorLeft();
      break;

    case 'R':
    case 'r':
      motorRight();
      break;

    case 'S':
    case 's':
      motorStop();
      break;

    case '\r':
    case '\n':
    case ' ':
    case '\t':
      break;

    default:
      Serial.print(F("UNKNOWN_COMMAND,"));
      Serial.println(command);
      printHelp();
      break;
  }
}

void printPotentiometer() {
  const int raw = analogRead(STEER_POT_PIN);

  Serial.print(F("POT,"));
  Serial.print(millis());
  Serial.print(',');
  Serial.print(raw);
  Serial.print(',');
  Serial.println(motorStateName());
}

void setup() {
  pinMode(STEER_MOTOR_IN1, OUTPUT);
  pinMode(STEER_MOTOR_IN2, OUTPUT);
  pinMode(STEER_POT_PIN, INPUT);

  Serial.begin(115200);

  motorStop();
  printHelp();

  lastPrintMs = millis();
}

void loop() {
  while (Serial.available() > 0) {
    const char command = static_cast<char>(Serial.read());
    processCommand(command);
  }

  const unsigned long now = millis();

  if (now - lastPrintMs >= PRINT_INTERVAL_MS) {
    lastPrintMs = now;
    printPotentiometer();
  }
}