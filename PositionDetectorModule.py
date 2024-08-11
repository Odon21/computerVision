# Name: Odon Crispo Cândido
# contact: anonymousmousecybersec@gmail.com
# YouTube: https://www.youtube.com/@hack-2173
#

import cv2
import mediapipe as mp
import time as t

landmark_color = (0, 0, 255)  
connection_color = (0, 255, 0)
landmark_thickness = 2  
connection_thickness = 2 
cTime = 0

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode, 
                                      model_complexity=1,
                                      smooth_landmarks=self.smooth, 
                                      min_detection_confidence=self.detectionCon, 
                                      min_tracking_confidence=self.trackCon)
        self.mp_draw = mp.solutions.drawing_utils
        
    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        if results.pose_landmarks:
            if draw:
                landmark_spec = self.mp_draw.DrawingSpec(color=landmark_color, thickness=landmark_thickness, circle_radius=2)
                connection_spec = self.mp_draw.DrawingSpec(color=connection_color, thickness=connection_thickness)
                self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_spec, connection_spec)
        return img


def main():
    pTime = 0
    # aqui pode alterar o parametro do VideoCapture() ::» 0 ou 1 é o indece da webCam
    # Ou pode alterar pela URL do video do esp32 cam
    cap = cv2.VideoCapture(0)
    detector = poseDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findPose(img)
        cTime = t.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Pose Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
