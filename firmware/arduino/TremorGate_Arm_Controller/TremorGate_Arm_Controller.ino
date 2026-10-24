#include <Servo.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// TremorGate Arm Controller
// Three-finger glove control
//
// Thumb  -> Waist    -> Joint 1 -> D10
// Index  -> Shoulder -> Joint 2 -> D9
// Middle -> Elbow    -> Joint 3 -> D8
//
// Joints 4-6 remain detached during three-joint glove control.
// Servo movement is gradual: 1 degree per 25 milliseconds.
// Startup is DISARMED.
// ============================================================

const uint8_t NUM_SERVOS = 6;

const uint8_t SERVO_PINS[NUM_SERVOS] = {
  10,  // Joint 1: Waist / Base
  9,   // Joint 2: Shoulder
  8,   // Joint 3: Elbow
  7,   // Joint 4: Wrist Roll
  6,   // Joint 5: Wrist Pitch
  5    // Joint 6: Gripper
};

const int MIN_ANGLE = 50;
const int MAX_ANGLE = 120;
const int CENTER_ANGLE = 90;

const unsigned long STEP_INTERVAL_MS = 25;

Servo servos[NUM_SERVOS];

int currentAngle[NUM_SERVOS];
int targetAngle[NUM_SERVOS];

bool armed = false;
bool moving = false;

unsigned long lastStepTime = 0;

const uint8_t COMMAND_LENGTH = 64;
char inputBuffer[COMMAND_LENGTH];
uint8_t inputIndex = 0;


// ============================================================
// Helper functions
// ============================================================

void updateMovingState() {
  moving = false;

  if (!armed) {
    return;
  }

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    if (servos[i].attached() &&
        currentAngle[i] != targetAngle[i]) {
      moving = true;
      return;
    }
  }
}


void detachAllServos() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    if (servos[i].attached()) {
      servos[i].detach();
    }
  }

  armed = false;
  moving = false;
}


void detachUnmappedServos() {
  // Preserve mapped joints 1-3 and 6; detach unused joints 4-5.
  for (uint8_t i = 3; i < 5; i++) {
    if (servos[i].attached()) {
      servos[i].detach();
    }
  }
}


void attachAllServos() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    if (!servos[i].attached()) {
      servos[i].attach(SERVO_PINS[i]);
      servos[i].write(currentAngle[i]);
    }
  }

  armed = true;
  updateMovingState();
}


void stopMovement() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    targetAngle[i] = currentAngle[i];
  }

  moving = false;

  Serial.println(
    F("STOP accepted. Targets set to current angles.")
  );

  Serial.println(
    F("Servo signals remain attached if armed.")
  );
}


void printStatus() {
  Serial.println();
  Serial.println(F("=== TremorGate Arm Status ==="));

  Serial.print(F("Controller: "));
  Serial.println(armed ? F("ARMED") : F("DISARMED"));

  Serial.print(F("Movement: "));
  Serial.println(moving ? F("MOVING") : F("IDLE"));

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    Serial.print(F("Joint "));
    Serial.print(i + 1);

    Serial.print(F(" | Pin D"));
    Serial.print(SERVO_PINS[i]);

    Serial.print(F(" | Current "));
    Serial.print(currentAngle[i]);

    Serial.print(F(" | Target "));
    Serial.print(targetAngle[i]);

    Serial.print(F(" | Servo "));
    Serial.println(
      servos[i].attached() ? F("ATTACHED") : F("DETACHED")
    );
  }

  Serial.println(F("============================="));
}


void printHelp() {
  Serial.println();
  Serial.println(F("=== TremorGate Commands ==="));
  Serial.println(F("STATUS       - Display controller status"));
  Serial.println(F("HELP         - Display commands"));
  Serial.println(F("TEST n angle - Attach and test a servo"));
  Serial.println(F("SET n angle  - Set a joint target"));
  Serial.println(F("STOP         - Stop commanded movement"));
  Serial.println(F("DISARM       - Detach all servo signals"));
  Serial.println(F("ARM          - Attach all six servos"));
  Serial.println();
  Serial.println(F("Joint 1: Waist, D10"));
  Serial.println(F("Joint 2: Shoulder, D9"));
  Serial.println(F("Joint 3: Elbow, D8"));
  Serial.println(F("Joint 4: Wrist Roll, D7"));
  Serial.println(F("Joint 5: Wrist Pitch, D6"));
  Serial.println(F("Joint 6: Gripper, D5"));
  Serial.println();
  Serial.print(F("Allowed angle range: "));
  Serial.print(MIN_ANGLE);
  Serial.print(F("-"));
  Serial.print(MAX_ANGLE);
  Serial.println(F(" degrees."));
  Serial.println(F("=========================="));
}


// ============================================================
// Set a target angle
// ============================================================

void setTarget(uint8_t index, int angle) {
  if (index >= NUM_SERVOS) {
    Serial.println(F("ERROR: Joint number must be 1-6."));
    return;
  }

  if (angle < MIN_ANGLE || angle > MAX_ANGLE) {
    Serial.print(F("ERROR: Angle must be between "));
    Serial.print(MIN_ANGLE);
    Serial.print(F(" and "));
    Serial.print(MAX_ANGLE);
    Serial.println(F(" degrees."));
    return;
  }

  targetAngle[index] = angle;

  Serial.print(F("Joint "));
  Serial.print(index + 1);
  Serial.print(F(" target set to "));
  Serial.print(angle);
  Serial.println(F(" degrees."));

  if (!armed) {
    Serial.println(
      F("Controller is DISARMED; no movement will occur.")
    );
    return;
  }

  if (!servos[index].attached()) {
    Serial.print(F("Joint "));
    Serial.print(index + 1);
    Serial.println(
      F(" is detached. Use TEST to attach it before movement.")
    );
    updateMovingState();
    return;
  }

  updateMovingState();
}


// ============================================================
// TEST command
//
// TEST 1 60  -> Attach waist
// TEST 2 60  -> Attach shoulder while preserving waist
// TEST 3 60  -> Attach elbow while preserving waist/shoulder
//
// For joints 1-3, previously attached mapped joints stay attached.
// Joints 4-6 are detached.
//
// TEST 4-6 retains single-servo test behaviour.
// ============================================================

void testServo(long jointNumber, long startAngle) {
  if (jointNumber < 1 || jointNumber > NUM_SERVOS) {
    Serial.println(F("ERROR: Joint number must be 1-6."));
    return;
  }

  if (startAngle < MIN_ANGLE || startAngle > MAX_ANGLE) {
    Serial.print(F("ERROR: Starting angle must be between "));
    Serial.print(MIN_ANGLE);
    Serial.print(F(" and "));
    Serial.print(MAX_ANGLE);
    Serial.println(F(" degrees."));
    return;
  }

  uint8_t index = (uint8_t)(jointNumber - 1);

  if ((jointNumber >= 1 && jointNumber <= 3) || jointNumber == 6) {

    // Keep the three mapped joints available together.
    // Ensure the other three servos remain detached.
    detachUnmappedServos();

    if (!servos[index].attached()) {
      currentAngle[index] = (int)startAngle;
      targetAngle[index] = (int)startAngle;

      servos[index].attach(SERVO_PINS[index]);
      servos[index].write((int)startAngle);
    } else {
      // Keep other mapped joints attached.
      currentAngle[index] = (int)startAngle;
      targetAngle[index] = (int)startAngle;

      servos[index].write((int)startAngle);
    }

    armed = true;
    updateMovingState();

    Serial.print(F("Mapped joint "));
    Serial.print(jointNumber);

    Serial.print(F(" attached on D"));
    Serial.print(SERVO_PINS[index]);

    Serial.print(F(" at "));
    Serial.print(startAngle);
    Serial.println(F(" degrees."));

    Serial.println(
      F("Other mapped joints remain attached; joints 4-5 detached.")
    );

    return;
  }

  // Joints 4-6 retain the original single-servo test behaviour.
  detachAllServos();

  currentAngle[index] = (int)startAngle;
  targetAngle[index] = (int)startAngle;

  servos[index].attach(SERVO_PINS[index]);
  servos[index].write((int)startAngle);

  armed = true;
  moving = false;

  Serial.print(F("Single-servo test: joint "));
  Serial.print(jointNumber);

  Serial.print(F(" on D"));
  Serial.print(SERVO_PINS[index]);

  Serial.print(F(", starting angle "));
  Serial.println(startAngle);
}


// ============================================================
// Process serial commands
// ============================================================

void processCommand(char *command) {
  // Convert command to uppercase.
  for (char *p = command; *p; p++) {
    *p = (char)toupper((unsigned char)*p);
  }

  // Remove leading whitespace.
  while (*command == ' ' || *command == '\t') {
    command++;
  }

  // Remove trailing whitespace.
  size_t length = strlen(command);

  while (length > 0 &&
         (command[length - 1] == ' ' ||
          command[length - 1] == '\t')) {
    command[length - 1] = '\0';
    length--;
  }

  if (length == 0) {
    return;
  }

  if (strcmp(command, "HELP") == 0) {
    printHelp();
    return;
  }

  if (strcmp(command, "STATUS") == 0) {
    printStatus();
    return;
  }

  if (strcmp(command, "DISARM") == 0) {
    detachAllServos();

    Serial.println(
      F("DISARMED: all servo signals detached.")
    );

    return;
  }

  if (strcmp(command, "STOP") == 0) {
    stopMovement();
    return;
  }

  if (strcmp(command, "ARM") == 0) {
    Serial.println(
      F("WARNING: attaching all six servos.")
    );

    Serial.println(
      F("Check mechanical clearance before using ARM.")
    );

    attachAllServos();

    Serial.println(
      F("ARMED at the controller's stored angles.")
    );

    return;
  }

  // TEST n angle
  if (strncmp(command, "TEST ", 5) == 0) {
    long jointNumber;
    long startAngle;
    char extra;

    int count = sscanf(
      command + 5,
      "%ld %ld %c",
      &jointNumber,
      &startAngle,
      &extra
    );

    if (count != 2) {
      Serial.println(
        F("Usage: TEST joint_number starting_angle")
      );
      return;
    }

    testServo(jointNumber, startAngle);
    return;
  }

  // SET n angle
  if (strncmp(command, "SET ", 4) == 0) {
    long jointNumber;
    long angle;
    char extra;

    int count = sscanf(
      command + 4,
      "%ld %ld %c",
      &jointNumber,
      &angle,
      &extra
    );

    if (count != 2) {
      Serial.println(F("Usage: SET joint angle"));
      return;
    }

    if (jointNumber < 1 || jointNumber > NUM_SERVOS) {
      Serial.println(F("ERROR: Joint number must be 1-6."));
      return;
    }

    setTarget((uint8_t)(jointNumber - 1), (int)angle);
    return;
  }

  Serial.print(F("Unknown command: "));
  Serial.println(command);
  Serial.println(F("Type HELP for available commands."));
}


// ============================================================
// Gradual servo movement
// Moves each attached servo by at most one degree per interval.
// ============================================================

void updateMotion() {
  if (!armed) {
    moving = false;
    return;
  }

  unsigned long now = millis();

  if (now - lastStepTime < STEP_INTERVAL_MS) {
    return;
  }

  lastStepTime = now;

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {

    // Never move a detached servo.
    if (!servos[i].attached()) {
      continue;
    }

    if (currentAngle[i] < targetAngle[i]) {
      currentAngle[i]++;
      servos[i].write(currentAngle[i]);
    }
    else if (currentAngle[i] > targetAngle[i]) {
      currentAngle[i]--;
      servos[i].write(currentAngle[i]);
    }
  }

  updateMovingState();
}


// ============================================================
// Setup
// ============================================================

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    currentAngle[i] = CENTER_ANGLE;
    targetAngle[i] = CENTER_ANGLE;
  }

  // No servo.attach() calls here.
  // Startup is intentionally DISARMED.
  // All servo signals remain detached.

  armed = false;
  moving = false;

  inputIndex = 0;

  Serial.println();
  Serial.println(F("===================================="));
  Serial.println(F("TremorGate Arm Controller"));
  Serial.println(F("Three-finger mapping firmware"));
  Serial.println(F("===================================="));

  Serial.println(F("Thumb  -> Waist    -> Joint 1 -> D10"));
  Serial.println(F("Index  -> Shoulder -> Joint 2 -> D9"));
  Serial.println(F("Middle -> Elbow    -> Joint 3 -> D8"));

  Serial.println(F("Startup state: DISARMED."));
  Serial.println(F("No servo signals have been attached."));

  printHelp();
}


// ============================================================
// Main loop
// ============================================================

void loop() {
  while (Serial.available() > 0) {
    char incoming = (char)Serial.read();

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      inputBuffer[inputIndex] = '\0';

      processCommand(inputBuffer);

      inputIndex = 0;
      continue;
    }

    if (inputIndex < COMMAND_LENGTH - 1) {
      inputBuffer[inputIndex++] = incoming;
    } else {
      inputIndex = 0;
      Serial.println(
        F("ERROR: Command too long. Input discarded.")
      );
    }
  }

  updateMotion();
}
