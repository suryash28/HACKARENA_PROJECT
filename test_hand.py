import cv2
import mediapipe as mp

print("Program Started")
print("Loading MediaPipe...")

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

print("MediaPipe Loaded Successfully")

# Open Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera Not Opened")
    exit()

print("✅ Camera Opened Successfully")

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Failed to Read Frame")
        break

    # Flip image for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("SignSpeak AI - Hand Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Program Closed")