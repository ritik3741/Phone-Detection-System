import cv2
from ultralytics import YOLO
import pygame
import time

# Initialize sound
pygame.mixer.init()
pygame.mixer.music.load("alert.mp3")

# Get alarm length
alarm_length = pygame.mixer.Sound("alert.mp3").get_length()

# Load YOLO
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

alarm_started = False
first_alarm_done = False
alarm_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    phone_found = False

    results = model(frame, stream=True)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone":
                phone_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE DETECTED",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

    current_time = time.time()

    # 🔊 Alarm control logic
    if phone_found:
        if not alarm_started:
            pygame.mixer.music.play()   # play ONCE
            alarm_start_time = current_time
            alarm_started = True
            first_alarm_done = False

        elif alarm_started and first_alarm_done:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)  # loop

    else:
        # Phone removed
        if alarm_started and not first_alarm_done:
            if current_time - alarm_start_time >= alarm_length:
                first_alarm_done = True
                pygame.mixer.music.stop()
                alarm_started = False
        elif alarm_started and first_alarm_done:
            pygame.mixer.music.stop()
            alarm_started = False

    # Check if first alarm finished
    if alarm_started and not first_alarm_done:
        if current_time - alarm_start_time >= alarm_length:
            first_alarm_done = True

    cv2.imshow("Phone Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
