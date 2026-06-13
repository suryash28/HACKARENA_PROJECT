import cv2
import mediapipe as mp
import pickle
import pyttsx3

# ======================
# Load Model
# ======================
with open("models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# ======================
# Text To Speech
# ======================
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# ======================
# MediaPipe
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ======================
# Variables
# ======================
sentence = []
current_prediction = "No Gesture"

# ======================
# Camera
# ======================
cap = cv2.VideoCapture(0)

print("=" * 50)
print("SignSpeak AI Started")
print("A = Add Current Word")
print("S = Speak Sentence")
print("C = Clear Sentence")
print("Q = Quit")
print("=" * 50)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            current_prediction = model.predict([landmarks])[0]

            probs = model.predict_proba([landmarks])[0]
            confidence = max(probs)

    else:
        current_prediction = "No Gesture"

    # ======================
    # Display Current Gesture
    # ======================
    cv2.putText(
        frame,
        f"Gesture: {current_prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # ======================
    # Sentence Display
    # ======================
    sentence_text = " ".join(sentence)

    cv2.putText(
        frame,
        "Sentence:",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        sentence_text,
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ======================
    # Instructions
    # ======================
    cv2.putText(
        frame,
        "A:Add  S:Speak  C:Clear  Q:Quit",
        (20, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2
    )

    cv2.imshow("SignSpeak AI", frame)

    # ======================
    # Keyboard Controls
    # ======================
    key = cv2.waitKey(1) & 0xFF

    # Add Word
    if key == ord('a'):

        if current_prediction != "No Gesture":

            if len(sentence) == 0 or sentence[-1] != current_prediction:

                sentence.append(current_prediction)

                print("Added:", current_prediction)

    # Speak Sentence
    elif key == ord('s'):

        if len(sentence) > 0:

            full_sentence = " ".join(sentence)

            print("Speaking:", full_sentence)

            try:
                engine.say(full_sentence)
                engine.runAndWait()
            except Exception as e:
                print("TTS Error:", e)

    # Clear Sentence
    elif key == ord('c'):

        sentence.clear()

        print("Sentence Cleared")

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()