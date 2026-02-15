✋ Gesture-Based Drone Control System (AI + Computer Vision)

A real-time AI-powered hand gesture recognition system that controls drone-like commands using a webcam. The project uses hand landmarks, machine learning, and live gesture classification to simulate intelligent drone control — including takeoff, landing, movement, speed control, and photo capture.

This system is built using computer vision + ML pipeline:

Dataset collection

Model training

Real-time gesture inference & command engine

🚀 Features

📷 Real-time hand tracking using MediaPipe

🧠 ML-based gesture classification (XGBoost)

🎮 Gesture → Command mapping

🔒 Safety lock system (Takeoff required before commands)

⚡ Dynamic speed control using finger distance

📸 Gesture-triggered photo capture

📊 Confidence filtering to avoid false commands

🧾 Custom dataset generation pipeline

🛠 Tech Stack

Python

OpenCV

MediaPipe Tasks API

NumPy

Pandas

Scikit-learn

XGBoost

Pickle (model persistence)

📌 Supported Gestures / Commands
Gesture Label	Command
UP	Move Up
DOWN	Move Down
LEFT	Move Left
RIGHT	Move Right
FORWARD	Move Forward
BACKWARD	Move Backward
BACKFLIP	Flip
HOVER	Hover
TAKEOFF	Unlock / Start
LAND	Stop / Lock
TAKE A PICTURE	Save frame
SPEED	Enter speed mode

📊 Model Details

Algorithm: XGBoost Classifier

Input Features: 63 landmark coordinates

Confidence Threshold: 95%

Low confidence → defaults to HOVER

Reduces false triggers

🎯 Key Innovations

End-to-end ML pipeline (data → training → inference)

Gesture confidence filtering

Dynamic speed control via geometry

Lock/unlock safety mechanism

Gesture-triggered camera system

Real-time visual feedback UI

Modular design for real drone integration

🏁 Use Cases

Gesture-based robotics control

Drone command systems

Touchless interfaces

Accessibility control systems

AI + CV hackathon demos

Smart surveillance control
