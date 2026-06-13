import cv2
import csv
import os

from src.components.lip_movement.lip_detector import LipDetector

detector = LipDetector()

label = input("Enter Word Label: ").strip().lower()

MAX_SAMPLES = 700
sample_count = 0

os.makedirs("data", exist_ok=True)

cap = cv2.VideoCapture(0)

with open("data/lip_data.csv", "a", newline="") as f:

    writer = csv.writer(f)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame, lips = detector.get_lip_landmarks(frame)

        if len(lips) > 0:

            row = []

            for p in lips:
                row.extend(p)

            row.append(label)

            writer.writerow(row)

            sample_count += 1

        cv2.putText(
            frame,
            f"{label.upper()} : {sample_count}/{MAX_SAMPLES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Lip Dataset Collection", frame)

        # Auto stop after target samples
        if sample_count >= MAX_SAMPLES:
            print(f"{MAX_SAMPLES} samples collected for '{label}'")
            break

        # Manual stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Collection Finished")