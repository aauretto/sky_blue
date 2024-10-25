// Define motor control pins
const int motor1Pin1 = 3;  // Motor 1 forward
const int motor1Pin2 = 4;  // Motor 1 backward
const int motor2Pin1 = 5;  // Motor 2 forward
const int motor2Pin2 = 6;  // Motor 2 backward

// Define speed variables
const int speedLeft = 150;   // Speed of the left motor (0-255)
const int speedRight = 200;  // Speed of the right motor (0-255)

void setup() {
  // Set motor control pins as outputs
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
}

void loop() {
  // Drive in a circle by running both motors at different speeds
  
  // Left motor (slower speed)
  analogWrite(motor1Pin1, speedLeft);  // Set speed for forward rotation
  digitalWrite(motor1Pin2, LOW);       // Ensure backward rotation is off

  // Right motor (faster speed)
  analogWrite(motor2Pin1, speedRight); // Set speed for forward rotation
  digitalWrite(motor2Pin2, LOW);       // Ensure backward rotation is off

  // Continue running indefinitely
}


