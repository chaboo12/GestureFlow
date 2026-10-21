import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ==========================================
# GESTUREFLOW SETTINGS
# ==========================================

screen_width, screen_height = pyautogui.size()

# Control area: smaller margins give a larger control area
FRAME_MARGIN_X = 5
FRAME_MARGIN_Y = 5

# Cursor smoothing: higher values respond faster
SMOOTHING = 0.15
DEAD_ZONE = 3
# Pinch detection with hysteresis
PINCH_START_RATIO = 0.30
PINCH_RELEASE_RATIO = 0.45

# Minimum time between clicks
CLICK_COOLDOWN = 0.45

# Camera settings
CAMERA_INDEX = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Keep PyAutoGUI from triggering its corner fail-safe
# Move the mouse to a screen corner to stop the program.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# ==========================================
# MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.6
)

detector = HandLandmarker.create_from_options(options)

# ==========================================
# WEBCAM SETUP
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    detector.close()
    raise RuntimeError(
        "Camera could not be opened. Check CAMERA_INDEX."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("GestureFlow started. Press Q to quit.")
print("Camera opened:", cap.isOpened())

# ==========================================
# CURSOR AND PINCH STATE
# ==========================================

previous_x, previous_y = pyautogui.position()

pinching = False
last_click_time = 0.0
frame_timestamp = 0

failed_reads = 0
MAX_FAILED_READS = 30

# ==========================================
# MAIN LOOP
# ==========================================

try:
    while True:

        success, frame = cap.read()

        if not success or frame is None or frame.size == 0:
            failed_reads += 1

            if failed_reads >= MAX_FAILED_READS:
                print("Camera stopped providing frames.")
                break

            time.sleep(0.03)
            continue

        failed_reads = 0

        # Mirror the camera for natural movement
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # ======================================
        # CONTROL AREA
        # ======================================

        left = FRAME_MARGIN_X
        top = FRAME_MARGIN_Y
        right = width - FRAME_MARGIN_X
        bottom = height - FRAME_MARGIN_Y

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (255, 255, 0),
            2
        )

        # ======================================
        # MEDIAPIPE IMAGE
        # ======================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # MediaPipe VIDEO timestamps must increase
        frame_timestamp = max(
            frame_timestamp + 1,
            time.monotonic_ns() // 1_000_000
        )

        results = detector.detect_for_video(
            mp_image,
            frame_timestamp
        )

        # ======================================
        # HAND DETECTION
        # ======================================

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]

            wrist = hand[0]
            thumb_tip = hand[4]
            index_tip = hand[8]
            middle_mcp = hand[9]

            # ----------------------------------
            # INDEX FINGER POSITION
            # ----------------------------------

            camera_x = int(index_tip.x * width)
            camera_y = int(index_tip.y * height)

            # ----------------------------------
            # MAP CAMERA POSITION TO SCREEN
            # ----------------------------------

            usable_width = max(1, right - left)
            usable_height = max(1, bottom - top)

            mapped_x = (camera_x - left) / usable_width
            mapped_y = (camera_y - top) / usable_height

            mapped_x = max(0.0, min(1.0, mapped_x))
            mapped_y = max(0.0, min(1.0, mapped_y))

            target_x = mapped_x * (screen_width - 1)
            target_y = mapped_y * (screen_height - 1)

            # ----------------------------------
            # STABLE CURSOR MOVEMENT
            # ----------------------------------

            # Calculate distance from the previous cursor position
            dx = target_x - previous_x
            dy = target_y - previous_y

            # Ignore tiny movements to reduce shaking
            if abs(dx) < DEAD_ZONE:
                dx = 0

            if abs(dy) < DEAD_ZONE:
                dy = 0

            # Smooth cursor movement
            current_x = previous_x + dx * SMOOTHING
            current_y = previous_y + dy * SMOOTHING

            # Keep the cursor within the screen
            current_x = max(0, min(screen_width - 1, current_x))
            current_y = max(0, min(screen_height - 1, current_y))

            # Move the cursor
            pyautogui.moveTo(
                int(current_x),
                int(current_y),
                duration=0
            )

            # Update previous position
            previous_x = current_x
            previous_y = current_y
            # ----------------------------------
            # LANDMARK POSITIONS
            # ----------------------------------

            thumb_x = int(thumb_tip.x * width)
            thumb_y = int(thumb_tip.y * height)

            # Draw index finger
            cv2.circle(
                frame,
                (camera_x, camera_y),
                8,
                (0, 255, 0),
                -1
            )

            # Draw pinch line
            cv2.line(
                frame,
                (camera_x, camera_y),
                (thumb_x, thumb_y),
                (255, 100, 0),
                2
            )

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                7,
                (255, 100, 0),
                -1
            )

            # ----------------------------------
            # NORMALIZED PINCH DISTANCE
            # ----------------------------------

            palm_size = math.hypot(
                wrist.x - middle_mcp.x,
                wrist.y - middle_mcp.y
            )

            if palm_size > 0.0001:

                finger_distance = math.hypot(
                    index_tip.x - thumb_tip.x,
                    index_tip.y - thumb_tip.y
                )

                pinch_ratio = finger_distance / palm_size

                current_time = time.monotonic()

                # ------------------------------
                # START PINCH: LEFT CLICK
                # ------------------------------

                if (
                    not pinching
                    and pinch_ratio < PINCH_START_RATIO
                ):

                    pinching = True

                    if (
                        current_time - last_click_time
                        >= CLICK_COOLDOWN
                    ):

                        pyautogui.click(button="left")

                        last_click_time = current_time
                        print("LEFT CLICK")

                # ------------------------------
                # RELEASE PINCH
                # ------------------------------

                elif (
                    pinching
                    and pinch_ratio > PINCH_RELEASE_RATIO
                ):

                    pinching = False

                # ------------------------------
                # DISPLAY STATUS
                # ------------------------------

                if pinching:
                    status = "LEFT CLICK"
                    status_color = (0, 255, 0)
                else:
                    status = "MOVE"
                    status_color = (255, 255, 255)

                cv2.putText(
                    frame,
                    status,
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Pinch: {pinch_ratio:.2f}",
                    (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # ----------------------------------
            # DRAW HAND LANDMARKS
            # ----------------------------------

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 255),
                    -1
                )

        else:

            # Release pinch state if hand disappears
            pinching = False

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ======================================
        # DISPLAY
        # ======================================

        cv2.imshow("GestureFlow", frame)

        # ======================================
        # EXIT
        # ======================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nGestureFlow stopped.")

except pyautogui.FailSafeException:
    print("\nGestureFlow stopped using the mouse-corner fail-safe.")

except Exception as error:
    print("GestureFlow error:", error)

finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Camera released.")