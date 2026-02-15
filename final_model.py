import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import os
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
]
if not os.path.exists("drone_photos"):
    os.makedirs("drone_photos")
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
with open('gesture_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model'] if isinstance(data, dict) else data
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
speed_meter_active = False
confirmed_speed = 50  
preview_speed = 50    
confirm_msg_timer = 0
try:
 while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy() 
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    current_time = time.time()
    key = cv2.waitKey(1) & 0xFF
    if is_flying:
        status_text, status_color = "ACTIVE: FLYING", (0, 255, 0)
    else:
        status_text, status_color = "LOCKED: Perform TAKEOFF", (0, 0, 255)
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        coords = []
        for p in lm: coords.extend([p.x, p.y, p.z])
        pred_num = model.predict([coords])[0]
        prediction = le.inverse_transform([pred_num])[0]
        probs = model.predict_proba([coords])[0]
        confidence = np.max(probs)
        if confidence < 0.95: 
            prediction = "HOVER"
        if prediction == "SPEED" and confidence > 0.95:
            speed_meter_active = True
        if speed_meter_active:
            current_display_cmd = "HOVER (SPEED MODE)"
            dist = math.sqrt((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)
            preview_speed = int(max(0, min(100, (dist - 0.05) * 400)))
            if key == ord('s'):
                confirmed_speed = preview_speed
                speed_meter_active = False 
                confirm_msg_timer = current_time
        else:
            current_display_cmd = prediction
            if current_time - lock_timer > 2.0:
                if not is_flying and prediction == "TAKEOFF":
                    is_flying, lock_timer = True, current_time
                elif is_flying and prediction == "LAND":
                    is_flying, lock_timer = False, current_time

            if is_flying and prediction == "TAKE A PICTURE":
                if current_time - last_photo_time > 3.0:
                    last_photo_time = current_time
                    cv2.imwrite(f"drone_photos/photo_{int(current_time)}.jpg", clean_frame)
                    photo_feedback_timer = current_time
        for conn in HAND_CONNECTIONS:
            cv2.line(frame, (int(lm[conn[0]].x * w), int(lm[conn[0]].y * h)), 
                     (int(lm[conn[1]].x * w), int(lm[conn[1]].y * h)), (0, 255, 0), 2)
    if speed_meter_active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(frame, f"CMD: {current_display_cmd} ", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"SETTING SPEED: {preview_speed}%", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'S' to Confirm", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        bar_color, bar_val = (0, 255, 255), preview_speed
    else:
        if is_flying:
            cv2.putText(frame, f"CMD: {current_display_cmd} ({confidence:.2f})", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"SPEED: {confirmed_speed}%", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        bar_color, bar_val = (255, 0, 0), confirmed_speed
    cv2.rectangle(frame, (50, 160), (250, 175), (50, 50, 50), -1)
    cv2.rectangle(frame, (50, 160), (50 + (bar_val * 2), 175), bar_color, -1)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    if current_time - confirm_msg_timer < 1.5:
        cv2.putText(frame, "SPEED CAPTURED!", (w//2-100, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if current_time - photo_feedback_timer < 1.0:
        cv2.putText(frame, "PHOTO SAVED!", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    cv2.imshow("Drone Controller", frame)
    if key == 27: break
finally:
 detector.close()
 cap.release()
 cv2.destroyAllWindows()
