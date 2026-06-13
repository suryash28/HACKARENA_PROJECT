import cv2
import mediapipe as mp

class LipDetector:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_lip_landmarks(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        lip_points = []

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            lips = [
                61,146,91,181,84,17,314,405,
                321,375,291,308,324,318,402,
                317,14,87,178,88,95
            ]

            h,w,_ = frame.shape

            for idx in lips:
                x = int(face_landmarks.landmark[idx].x*w)
                y = int(face_landmarks.landmark[idx].y*h)

                lip_points.append([x,y])

                cv2.circle(frame,(x,y),2,(0,255,0),-1)

        return frame, lip_points