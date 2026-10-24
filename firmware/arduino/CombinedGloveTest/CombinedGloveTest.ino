#include <Wire.h>


const uint8_t IMU_ADDR = 0x68;
const uint8_t flexPins[4] = {A0, A1, A2, A3};
const uint32_t SAMPLE_INTERVAL_US = 10000;

uint32_t nextSampleTime;
uint32_t startTime;
bool imuReady = false;

bool writeIMU(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readIMU(uint8_t reg, uint8_t *buffer, uint8_t length) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;

  uint8_t received = Wire.requestFrom(IMU_ADDR, length, (uint8_t)true);
  if (received != length) {
    while (Wire.available()) Wire.read();
    return false;
  }

  for (uint8_t i = 0; i < length; i++) buffer[i] = Wire.read();
  return true;
}

bool initializeIMU() {
  uint8_t identity;
  if (!readIMU(0x75, &identity, 1)) {
    Serial.println("# ERROR: IMU identity read failed");
    return false;
  }

  Serial.print("# IMU identity: 0x");
  Serial.println(identity, HEX);

  if (!writeIMU(0x6B, 0x00)) return false;
  delay(100);

  if (!writeIMU(0x1C, 0x00) || !writeIMU(0x1B, 0x00)) return false;

  Serial.println("# IMU initialized");
  return true;
}

void setup() {
  Serial.begin(115200);  // USB debugging/upload remains available
  Wire.begin();
  Wire.setClock(100000);
  delay(1000);

  Serial.println("# TremorGate four-flex + IMU USB stream");
  imuReady = initializeIMU();
  Serial.println("time_ms,thumb,index,middle,ring,ax,ay,az,gx,gy,gz,imu_ok");

  startTime = millis();
  nextSampleTime = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextSampleTime) < 0) return;
  nextSampleTime += SAMPLE_INTERVAL_US;

  uint16_t flex[4];
  for (uint8_t i = 0; i < 4; i++) flex[i] = analogRead(flexPins[i]);

  uint8_t raw[14];
  bool valid = imuReady && readIMU(0x3B, raw, 14);

  Serial.print(millis() - startTime);
  for (uint8_t i = 0; i < 4; i++) {
    Serial.print(',');
    Serial.print(flex[i]);
  }

  Serial.print(',');
  if (valid) {
    for (uint8_t i = 0; i < 3; i++) {
      int16_t value = (int16_t)(((uint16_t)raw[i * 2] << 8) | raw[i * 2 + 1]);
      Serial.print(value);
      Serial.print(',');
    }

    for (uint8_t i = 0; i < 3; i++) {
      uint8_t offset = 8 + i * 2;
      int16_t value = (int16_t)(((uint16_t)raw[offset] << 8) | raw[offset + 1]);
      Serial.print(value);
      if (i < 2) Serial.print(',');
    }
  } else {
    Serial.print(",,,,,");
  }

  Serial.print(',');
  Serial.println(valid ? 1 : 0);
}
