# HAPTIC ALERT ENCHANCEMENT SYSTEM
## Overview
This code identifies vehicle in a lane and send messages based on their proximity to arduino for which arduino simply sets vibtration according to the message recieved

## Features
* Custom Fine tuned YOLO is used for detecting vehciles
* Supervision is used to track vehicles and get the speed 
* MTTQ is used to send the message from system to arduino
* Raspberry Pi is the hardware used for computation

## Dependenciens
* opencv
* pandas
* supervision
* os 
* json
* socket
* paho
* numpy
* inference
* collection
* time