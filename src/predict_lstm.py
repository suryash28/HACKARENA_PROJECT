import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3

# =====================================
# LOAD MODEL
# =====================================

print("Loading Model...")

model = tf.keras.models.load_model(
    "models/gesture_lstm.keras"
)

with open(
    "models/label_encoder.pkl",
    "rb"
) as f:
    label_encoder = pickle.load(f)

print("\nClasses:")
print(label_encoder.classes_)

# =====================================
# TEXT TO SPEECH
# =====================================

engine = pyttsx3.init()
engine.setProperty('rate', 150)

# =====================================
# MEDIAPIPE
# =====================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

sequence = []

current_prediction = "UNKNOWN"
confidence = 0.0

sentence = []

# =====================================
# MAIN LOOP
# =====================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:

                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            sequence.append(landmarks)

            # Keep only last 30 frames
            sequence = sequence[-30:]

            if len(sequence) == 30:

                input_data = np.expand_dims(
                    sequence,
                    axis=0
                )

                prediction = model.predict(
                    input_data,
                    verbose=0
                )[0]

                confidence = float(
                    np.max(prediction)
                )

                class_id = int(
                    np.argmax(prediction)
                )

                print("\n========================")

                for i, label in enumerate(
                    label_encoder.classes_
                ):
                    print(
                        f"{label:<15} : {prediction[i]:.4f}"
                    )

                print("------------------------")

                print(
                    "BEST:",
                    label_encoder.classes_[class_id],
                    "| CONF:",
                    confidence
                )

                # Confidence threshold

                if confidence > 0.85:

                    current_prediction = (
                        label_encoder.classes_[class_id]
                    )

                else:

                    current_prediction = (
                        "UNKNOWN"
                    )

    else:

        current_prediction = "NO HAND"

    # =====================================
    # DISPLAY
    # =====================================

    if current_prediction == "UNKNOWN":

        color = (0, 0, 255)

    elif current_prediction == "NO HAND":

        color = (0, 255, 255)

    else:

        color = (0, 255, 0)

    cv2.putText(
        frame,
        f"{current_prediction}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    sentence_text = " ".join(sentence)

    cv2.putText(
        frame,
        "Sentence:",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        sentence_text,
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Add Word",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        "S = Speak",
        (20, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        "C = Clear",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.imshow(
        "VoiceBeyond AI",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # =====================================
    # ADD WORD
    # =====================================

    if key == ord(' '):

        if (
            current_prediction != "UNKNOWN"
            and
            current_prediction != "NO HAND"
        ):

            sentence.append(
                current_prediction
            )

            print(
                "\nAdded:",
                current_prediction
            )

            sequence = []

    # =====================================
    # SPEAK SENTENCE
    # =====================================

    elif key == ord('s'):

        if len(sentence) > 0:

            text = " ".join(sentence)

            print(
                "\nSpeaking:",
                text
            )

            try:

                engine.say(text)
                engine.runAndWait()

            except Exception as e:

                print(
                    "TTS Error:",
                    e
                )

    # =====================================
    # CLEAR SENTENCE
    # =====================================

    elif key == ord('c'):

        sentence = []

        print(
            "\nSentence Cleared"
        )

    # =====================================
    # QUIT
    # =====================================

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()