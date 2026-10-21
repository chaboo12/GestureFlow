import cv2
import time

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

print("Camera opened successfully")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)

# Discard initial frames
for i in range(10):
    cap.grab()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame read failed")
        time.sleep(0.1)
        continue

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()