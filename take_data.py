import cv2
import csv
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model_path = r"C:\Users\SARTHAK\OneDrive\Desktop\drone\hand_landmarker.task"
csv_file = "drone_dataset.csv"
KEY_MAP = {
    ord('u'): "UP",
    ord('d'): "DOWN",
    ord('l'): "LEFT",
    ord('r'): "RIGHT",
    ord('f'): "FORWARD",
    ord('b'): "BACKWARD",
    ord('x'): "BACKFLIP",
    ord('h'): "HOVER",
    ord('t'): "TAKEOFF",
    ord('p'): "TAKE A PICTURE",
    ord('v'): "LAND",
    ord('s'): "SPEED"
}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'c{i}' for i in range(63)] + ['label']
        writer.writerow(header)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    running_mode=vision.RunningMode.VIDEO, 
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
saved_count = 0
print("U=Up, D=Down, L=Left, R=Right, F=Forward, B=Backward, X=Flip, H=Hover, T=Takeoff, P=TAKE A PICTURE, V=LAND")
while cap.isOpened():   
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    key = cv2.waitKey(1) & 0xFF
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        for conn in HAND_CONNECTIONS:
            start = lm[conn[0]]
            end = lm[conn[1]]
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 0, 255), -1)
        if key in KEY_MAP:
            label = KEY_MAP[key]
            row = []
            for p in lm: row.extend([p.x, p.y, p.z])
            row.append(label)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush() 
            saved_count += 1
            print(f"Recorded: {label} | Total: {saved_count}")
    cv2.putText(frame, f"SAMPLES: {saved_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Speed Collector - Connected Dots", frame)
    if key == 27: break
cap.release()
cv2.destroyAllWindows()
