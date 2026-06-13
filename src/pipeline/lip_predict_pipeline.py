import cv2
import joblib
import numpy as np
from collections import Counter

from src.components.lip_movement.lip_detector import LipDetector

model = joblib.load("models/lip_model.pkl")

detector = LipDetector()

cap = cv2.VideoCapture(0)

predictions = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame, lips = detector.get_lip_landmarks(frame)

    display_text = "Waiting..."

    if len(lips) > 15:

        upper_lip = lips[5]
        lower_lip = lips[15]

        mouth_open = abs(upper_lip[1] - lower_lip[1])

        # Increase threshold to avoid random predictions
        if mouth_open > 15:

            row = []

            for p in lips:
                row.extend(p)

            row = np.array(row, dtype=float)

            # Same preprocessing used during training
            row = row - np.mean(row)

            probs = model.predict_proba([row])[0]

            confidence = np.max(probs)

            pred = model.classes_[np.argmax(probs)]

            # Show debug confidence
            print(
                f"Prediction={pred} | Confidence={confidence:.2f}"
            )

            # Strong confidence only
            if confidence > 0.80:

                predictions.append(pred)

                # Keep last 20 predictions
                if len(predictions) > 20:
                    predictions.pop(0)

                if len(predictions) >= 15:

                    label, count = Counter(
                        predictions
                    ).most_common(1)[0]

                    # Require strong majority
                    if count >= 12:

                        display_text = (
                            f"{label} ({confidence:.2f})"
                        )

        else:
            predictions.clear()

    cv2.putText(
        frame,
        display_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Lip Movement Prediction",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()