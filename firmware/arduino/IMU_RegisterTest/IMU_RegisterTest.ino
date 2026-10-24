#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;

bool readRegister(uint8_t reg, uint8_t &value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  uint8_t count = Wire.requestFrom(
    (uint8_t)MPU_ADDR,
    (uint8_t)1,
    (uint8_t)true
  );

  if (count != 1 || !Wire.available()) {
    return false;
  }

  value = Wire.read();
  return true;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(100000);
  delay(1000);

  const uint8_t registers[] = {
    0x75, 0x6B, 0x1B, 0x1C, 0x3B, 0x3D, 0x3F, 0x41, 0x43
  };

  Serial.println("Register diagnostic");

  for (uint8_t i = 0; i < sizeof(registers); i++) {
    uint8_t value = 0;

    Serial.print("Register 0x");
    Serial.print(registers[i], HEX);
    Serial.print(": ");

    if (readRegister(registers[i], value)) {
      Serial.print("0x");
      if (value < 16) Serial.print("0");
      Serial.println(value, HEX);
    } else {
      Serial.println("READ FAILED");
    }
  }
}

void loop() {
  delay(1000);
}
