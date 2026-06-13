import cv2
import mediapipe as mp
import numpy as np
import os

gesture_name = input("Enter Gesture Name: ").upper()

DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30

os.makedirs(os.path.join(DATA_PATH, gesture_name), exist_ok=True)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sequence_count = len(os.listdir(os.path.join(DATA_PATH, gesture_name)))

recording = False
frames = []

print("R = Record Gesture")
print("Q = Quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    landmarks = []

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y

        for lm in hand_landmarks.landmark:

            landmarks.extend([
                lm.x - base_x,
                lm.y - base_y,
                lm.z
            ])

    else:
        landmarks = [0] * 63

    if recording:

        frames.append(landmarks)

        cv2.putText(
            frame,
            f"Recording {len(frames)}/{SEQUENCE_LENGTH}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if len(frames) == SEQUENCE_LENGTH:

            sequence_count += 1

            save_path = os.path.join(
                DATA_PATH,
                gesture_name,
                f"{sequence_count}.npy"
            )

            np.save(save_path, np.array(frames))

            print("Saved:", save_path)

            frames = []
            recording = False

    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        "R=Record Q=Quit",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.imshow("Sequence Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        recording = True
        frames = []
        print("Recording Started...")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()