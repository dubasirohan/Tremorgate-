#include <Wire.h>

void setup() {
  Serial.begin(115200);
  delay(1500);

  Wire.begin();
  Wire.setClock(100000);

  Serial.println();
  Serial.println("TremorGate: I2C Scanner");
  Serial.println("Scanning hardware I2C bus...");

  int found = 0;

  for (uint8_t address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    uint8_t error = Wire.endTransmission();

    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
      found++;
    }
  }

  if (found == 0) {
    Serial.println("No I2C devices found.");
  } else {
    Serial.print("Total devices found: ");
    Serial.println(found);
  }
}

void loop() {
}
