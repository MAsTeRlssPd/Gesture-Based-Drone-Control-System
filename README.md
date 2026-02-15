# ✈️ Gesture-Based Drone Control System

A real-time hand gesture recognition system that enables touchless drone control using a webcam, hand landmarks, and a machine learning classifier. The system detects hand gestures and maps them to drone-like commands such as takeoff, land, movement, hover, flip, speed control, and photo capture.

------------------------------------------------------------

## FEATURES

- Real-time hand tracking with landmark detection
- Custom gesture dataset collection pipeline
- Lightweight ML classification using landmark features (63 values)
- XGBoost-based gesture model
- Label encoder + model persistence
- Real-time gesture inference
- Lock / Unlock flight safety system
- Confidence threshold filtering
- Gesture-triggered photo capture
- Gesture-based speed control mode
- Visual hand skeleton overlay
- Command cooldown and debounce timers
- CSV dataset auto creation

------------------------------------------------------------

## SUPPORTED COMMANDS

UP – Move up  
DOWN – Move down  
LEFT – Move left  
RIGHT – Move right  
FORWARD – Move forward  
BACKWARD – Move backward  
BACKFLIP – Flip  
HOVER – Stay in place  
TAKEOFF – Activate control  
LAND – Lock system  
TAKE A PICTURE – Capture image  
SPEED – Enter speed control mode  

------------------------------------------------------------

## HOW THE SYSTEM WORKS

### 1. Hand Landmark Detection
- 21 hand landmarks detected per frame
- Each landmark has x, y, z coordinates
- Total features per sample = 63

### 2. Dataset Collection (take_data.py)
- Opens webcam
- Detects hand landmarks
- Draws skeleton
- Saves 63 landmark values + label to CSV
- Labels assigned using keyboard keys
- Fast sample recording mode

### 3. Model Training (trainer.py)
- Reads drone_dataset.csv
- Encodes labels
- Splits train/test data
- Trains XGBoost classifier
- Saves:
  - gesture_model.pkl
  - label_encoder.pkl

### 4. Real-Time Controller (final_model.py)
- Loads trained model and label encoder
- Runs live webcam detection
- Predicts gesture per frame
- Applies confidence threshold
- Controls system state
- Handles speed mode and photo capture
- Displays UI feedback

------------------------------------------------------------

## PROJECT STRUCTURE

take_data.py          Dataset collection  
trainer.py            Model training  
final_model.py        Real-time controller  
gesture_model.pkl     Trained model  
label_encoder.pkl     Label encoder  
hand_landmarker.task  Landmark model file  
drone_dataset.csv     Dataset  
drone_photos/         Saved images  

------------------------------------------------------------

## REQUIREMENTS

Install dependencies:

pip install opencv-python mediapipe numpy pandas scikit-learn xgboost

------------------------------------------------------------

## SETUP STEPS

1. Place hand_landmarker.task in project folder.

2. Collect Data:

python take_data.py

Key bindings:
U D L R F B X H T P V S  
ESC to exit

3. Train Model:

python trainer.py

4. Run Controller:

python final_model.py

------------------------------------------------------------

## SAFETY LOGIC

- System starts LOCKED
- TAKEOFF gesture unlocks control
- LAND gesture locks control
- 2-second lock timer prevents rapid toggling
- Low confidence predictions fallback to HOVER
- No-hand detected while flying → hover behavior

------------------------------------------------------------

## PHOTO CAPTURE

Gesture: TAKE A PICTURE

- Saves clean frame
- Stored in drone_photos folder
- 3-second cooldown
- On-screen confirmation message

------------------------------------------------------------

## SPEED CONTROL MODE

Gesture: SPEED

- Activates speed meter mode
- Thumb–index finger distance sets speed
- Press S key to confirm
- Visual speed bar shown
- Stored as confirmed_speed percentage

------------------------------------------------------------

## KEY TAKEAWAYS

- Built full gesture AI pipeline
- Used landmark-based ML instead of heavy image CNN
- Designed real-time safety state machine
- Added confidence filtering
- Implemented human-machine interaction feedback
- Created extensible command architecture
- Ready for real drone SDK integration
- Hackathon and portfolio ready

------------------------------------------------------------

## AUTHOR

Sarthak Tomar
