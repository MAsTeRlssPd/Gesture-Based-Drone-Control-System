import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]

if not os.path.exists("drone_photos"):
    os.makedirs("drone_photos")
with open('gesture_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model'] if isinstance(data, dict) else data

# 2. Setup MediaPipe
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    running_mode=vision.RunningMode.VIDEO, 
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

is_flying = False       
lock_timer = 0          
last_photo_time = 0     
photo_feedback_timer = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy() # Backup for clean photos
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    current_time = time.time()

    if is_flying:
        status_text = "ACTIVE: FLYING"
        status_color = (0, 255, 0)
    else:
        status_text = "LOCKED: Perform TAKEOFF"
        status_color = (0, 0, 255) 

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        # Draw Skeleton
        for conn in HAND_CONNECTIONS:
            pt1 = (int(lm[conn[0]].x * w), int(lm[conn[0]].y * h))
            pt2 = (int(lm[conn[1]].x * w), int(lm[conn[1]].y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Get Prediction
        coords = []
        for p in lm: coords.extend([p.x, p.y, p.z])
        prediction = model.predict([coords])[0]
        probs = model.predict_proba([coords])[0]
        confidence = np.max(probs)

        if confidence < 0.7: prediction = "HOVER"

        # 1. Takeoff / Land Toggle
        if current_time - lock_timer > 2.0:
            if not is_flying and prediction == "TAKEOFF" and confidence > 0.8:
                is_flying = True
                lock_timer = current_time
            elif is_flying and prediction == "LAND" and confidence > 0.8:
                is_flying = False
                lock_timer = current_time

        # 2. Camera Logic (Only when flying)
        if is_flying:
            if prediction == "TAKE A PICTURE" and confidence > 0.8:
                if current_time - last_photo_time > 3.0:
                    last_photo_time = current_time
                    filename = f"drone_photos/photo_{int(current_time)}.jpg"
                    cv2.imwrite(filename, clean_frame) # Saving the clean version
                    photo_feedback_timer = current_time

            # Show current active command
            cv2.putText(frame, f"CMD: {prediction}", (50, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    else:
        if is_flying:
            cv2.putText(frame, "CMD: HOVER (No Hand)", (50, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # UI Feedback for Photo
    if current_time - photo_feedback_timer < 1.0:
        cv2.putText(frame, "PHOTO SAVED!", (w//2 - 100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow("Drone Control", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()

cv2.destroyAllWindows()
