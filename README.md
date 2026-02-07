# ✈️ Gesture-Based Drone Control System

A real-time hand gesture recognition system that enables **touchless drone control** using a webcam, MediaPipe hand landmarks, and a machine learning classifier. The system detects hand gestures and maps them to drone-like commands such as **takeoff, land, movement, hover, flip, and photo capture**.

This project includes:

- 📊 Dataset collection pipeline  
- 🧠 ML gesture classification model usage  
- 🎥 Real-time gesture inference and command execution  

---

## 🚀 Features

- Real-time hand tracking using MediaPipe Hand Landmarker
- Gesture-based command recognition
- Custom dataset collection tool
- ML model prediction with confidence filtering
- Safety lock system (Takeoff → Active → Land)
- Gesture-triggered photo capture
- Visual skeleton overlay for hand landmarks
- Command cooldown & debounce logic
- CSV dataset auto-builder

---

## 🎮 Supported Commands

| Gesture Label | Action |
|----------------|----------|
| UP | Move up |
| DOWN | Move down |
| LEFT | Move left |
| RIGHT | Move right |
| FORWARD | Move forward |
| BACKWARD | Move backward |
| BACKFLIP | Perform flip |
| HOVER | Stay in place |
| TAKEOFF | Activate control |
| LAND | Lock system |
| TAKE A PICTURE | Capture frame |

---
