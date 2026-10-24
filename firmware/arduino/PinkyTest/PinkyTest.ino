const int PINKY_PIN = A4;

void setup() {
  Serial.begin(115200);
}

void loop() {
  long total = 0;

  for (int i = 0; i < 10; i++) {
    total += analogRead(PINKY_PIN);
    delay(5);
  }

  float average = total / 10.0;

  Serial.print("Pinky ADC: ");
  Serial.println(average);

  delay(200);
}
