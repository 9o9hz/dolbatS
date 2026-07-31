void setup() {
  pinMode(A4, INPUT);
  Serial.begin(115200);

}

void loop() {
  Serial.println(analogRead(A4));

}
