#include <Servo.h>

Servo waist;

const int WAIST_PIN = 5;

void setup() {
  Serial.begin(115200);
  waist.attach(WAIST_PIN);
  waist.write(90);
  delay(1000);
  Serial.println("Waist servo test ready.");
  Serial.println("Enter an angle between 80 and 100.");
}

void loop() {
  if (Serial.available()) {
    int angle = Serial.parseInt();

    while (Serial.available()) {
      Serial.read();
    }

    if (angle >= 80 && angle <= 100) {
      waist.write(angle);
      Serial.print("Waist angle: ");
      Serial.println(angle);
    } else {
      Serial.println("Enter an angle between 80 and 100.");
    }
  }
}
