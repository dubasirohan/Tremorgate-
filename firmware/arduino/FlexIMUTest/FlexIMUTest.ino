#include <Wire.h>

const uint8_t flexPins[4] = {A0, A1, A2, A3};
const uint8_t MPU_ADDR = 0x68;

bool imuReady = false;

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readRegisters(uint8_t reg, uint8_t count, uint8_t *data) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  uint8_t received = Wire.requestFrom(MPU_ADDR, count);

  if (received != count) {
    return false;
  }

  for (uint8_t i = 0; i < count; i++) {
    if (!Wire.available()) {
      return false;
    }
    data[i] = Wire.read();
  }

  return true;
}

int16_t combineBytes(uint8_t highByte, uint8_t lowByte) {
  return (int16_t)(((uint16_t)highByte << 8) | lowByte);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin();
  Wire.setClock(100000);

  uint8_t identity = 0;

  if (readRegisters(0x75, 1, &identity) && identity == 0x68) {
    bool wakeOK = writeRegister(0x6B, 0x00);
    delay(100);

    bool accelOK = writeRegister(0x1C, 0x00);
    bool gyroOK = writeRegister(0x1B, 0x00);

    imuReady = wakeOK && accelOK && gyroOK;
  }

  Serial.println("# TremorGate hardware-I2C test");
  Serial.print("# MPU6050 status: ");
  Serial.println(imuReady ? "READY" : "NOT_DETECTED_OR_CONFIG_FAILED");

  Serial.println("time_ms,thumb,index,middle,ring,ax,ay,az,gx,gy,gz");
}

void loop() {
  int flex[4];

  for (uint8_t i = 0; i < 4; i++) {
    long total = 0;

    for (uint8_t j = 0; j < 5; j++) {
      total += analogRead(flexPins[i]);
      delayMicroseconds(500);
    }

    flex[i] = total / 5;
  }

  int16_t ax = 0, ay = 0, az = 0;
  int16_t gx = 0, gy = 0, gz = 0;

  uint8_t raw[14];
  bool sampleOK = imuReady && readRegisters(0x3B, 14, raw);

  if (sampleOK) {
    ax = combineBytes(raw[0], raw[1]);
    ay = combineBytes(raw[2], raw[3]);
    az = combineBytes(raw[4], raw[5]);

    gx = combineBytes(raw[8], raw[9]);
    gy = combineBytes(raw[10], raw[11]);
    gz = combineBytes(raw[12], raw[13]);
  }

  Serial.print(millis());

  for (uint8_t i = 0; i < 4; i++) {
    Serial.print(',');
    Serial.print(flex[i]);
  }

  if (sampleOK) {
    Serial.print(',');
    Serial.print(ax);
    Serial.print(',');
    Serial.print(ay);
    Serial.print(',');
    Serial.print(az);
    Serial.print(',');
    Serial.print(gx);
    Serial.print(',');
    Serial.print(gy);
    Serial.print(',');
    Serial.println(gz);
  } else {
    Serial.println(",,,,,,");
  }

  delay(30);
}
