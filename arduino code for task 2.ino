const int alarmPin = 8;
const int ledPin = 13;
#include <LiquidCrystal.h>
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);


void setup() {
  lcd.begin(16, 2);
  lcd.setCursor(0, 0);
  pinMode(alarmPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char signal = Serial.read();
    if (signal == '1') {
      lcd.print("Be attention");
      digitalWrite(alarmPin, HIGH);  // Turn on the alarm
      digitalWrite(ledPin, HIGH);  // Turn on the led
      delay(100);  // Keep it on for 5 seconds
      digitalWrite(alarmPin, LOW);  // Turn off the alarm
      digitalWrite(ledPin, LOW);    // Turn off the led
       lcd.clear();
    }
  }
}