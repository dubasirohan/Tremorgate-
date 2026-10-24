/*
TremorGate ESP32 firmware.
CSV contract:
timestamp_ms,flex1,flex2,flex3,flex4,flex5,ax_g,ay_g,az_g,gx_rad_s,gy_rad_s,gz_rad_s

Wiring:
- Flex sensors: voltage divider, fixed 10 kOhm resistor to GND, flex sensor to 3.3 V.
  Junction -> ADC pins 34,35,32,33,36 (ADC1 pins; input-only is OK).
  IMPORTANT: verify your exact ESP32 board pinout; do not apply >3.3 V.
- MPU6050: VCC 3.3 V, GND GND, SDA GPIO21, SCL GPIO22 (default ESP32 I2C).
  AD0 low => address 0x68.
*/
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;
const int FLEX[5] = {34,35,32,33,36};
const uint32_t PERIOD_US=10000;
uint32_t next_us=0;

void setup(){
  Serial.begin(115200);
  Wire.begin(21,22);
  if(!mpu.begin(0x68)){
    Serial.println("# MPU6050 not found");
    while(true){delay(1000);}
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
  for(int i=0;i<5;i++) pinMode(FLEX[i],INPUT);
  next_us=micros();
}

void loop(){
  uint32_t now=micros();
  if((int32_t)(now-next_us)<0) return;
  next_us += PERIOD_US;
  sensors_event_t a,g,temp;
  mpu.getEvent(&a,&g,&temp);
  Serial.print(millis());
  for(int i=0;i<5;i++){Serial.print(',');Serial.print(analogRead(FLEX[i]));}
  Serial.print(',');Serial.print(a.acceleration.x/9.80665f,6);
  Serial.print(',');Serial.print(a.acceleration.y/9.80665f,6);
  Serial.print(',');Serial.print(a.acceleration.z/9.80665f,6);
  Serial.print(',');Serial.print(g.gyro.x,6);
  Serial.print(',');Serial.print(g.gyro.y,6);
  Serial.print(',');Serial.println(g.gyro.z,6);
}
