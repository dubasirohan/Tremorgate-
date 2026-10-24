const int flexPins[4] = {A0, A1, A2, A3};

void setup() {
  Serial.begin(115200);
  Serial.println("time_ms,thumb,index,middle,ring");
}

void loop() {
  Serial.print(millis());

  for (int i = 0; i < 4; i++) {
    Serial.print(",");
    Serial.print(analogRead(flexPins[i]));
  }

  Serial.println();
  delay(20);
}
