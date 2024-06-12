void setup() {
  Serial.begin(9600);
}
void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    Serial.print("You sent me: ");
    Serial.println(data);
  }
}



int in1 = 8;
int in2 = 9;

void setup() 
{
  Serial.begin(9600);
 pinMode(in1, OUTPUT);
 pinMode(in2, OUTPUT);
  Serial.println("ready");

}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    Serial.print("command: ");
    Serial.println(command);

    if (command == 'F') {

      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
      delay(2000);

    }else if (command == 'B') {

      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);

        }else if (command == 'S') {

      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
    }
    }
  }
