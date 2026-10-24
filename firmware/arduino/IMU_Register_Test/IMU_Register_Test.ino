#include <Wire.h>

uint8_t imuAddress = 0x68;

bool readRegister(uint8_t reg, uint8_t &value) {
  Wire.beginTransmission(imuAddress);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(imuAddress, (uint8_t)1) != 1) return false;

  value = Wire.read();
  return true;
}

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(imuAddress);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(100000);
  delay(500);

  Serial.println("\n=== IMU Register Diagnostic ===");

  bool found = false;

  for (uint8_t addr = 0x68; addr <= 0x69; addr++) {
    Wire.beginTransmission(addr);

    if (Wire.endTransmission() == 0) {
      imuAddress = addr;
      found = true;
      break;
    }
  }

  if (!found) {
    Serial.println("ERROR: No I2C device at 0x68 or 0x69.");
    return;
  }

  Serial.print("I2C device found at 0x");
  Serial.println(imuAddress, HEX);

  uint8_t identity;

  if (readRegister(0x75, identity)) {
    Serial.print("WHO_AM_I = 0x");
    Serial.println(identity, HEX);
  } else {
    Serial.println("ERROR: Cannot read WHO_AM_I.");
    return;
  }

  if (writeRegister(0x6B, 0x00)) {
    Serial.println("Wake-up register write succeeded.");
  } else {
    Serial.println("ERROR: Wake-up register write failed.");
    return;
  }

  delay(100);

  for (int i = 0; i < 10; i++) {
    uint8_t b[6];
    bool ok = true;

    Wire.beginTransmission(imuAddress);
    Wire.write(0x3B);

    if (Wire.endTransmission(false) != 0) {
      ok = false;
    } else if (Wire.requestFrom(imuAddress, (uint8_t)6) != 6) {
      ok = false;
    } else {
      for (int j = 0; j < 6; j++) b[j] = Wire.read();
    }

    if (ok) {
      int16_t ax = (int16_t)((b[0] << 8) | b[1]);
      int16_t ay = (int16_t)((b[2] << 8) | b[3]);
      int16_t az = (int16_t)((b[4] << 8) | b[5]);

      Serial.print("Accel: ");
      Serial.print(ax);
      Serial.print(", ");
      Serial.print(ay);
      Serial.print(", ");
      Serial.println(az);
    } else {
      Serial.println("ERROR: Accelerometer read failed.");
    }

    delay(300);
  }

  Serial.println("Diagnostic complete.");
}

void loop() {}
