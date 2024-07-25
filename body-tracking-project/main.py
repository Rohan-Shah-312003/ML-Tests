# import cv2
# import mediapipe as mp
# import time
# import threading

# class poseDetector:
#     def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon
#         self.pTime = 0

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(static_image_mode=self.mode,
#                                     smooth_landmarks=self.smooth,
#                                     min_detection_confidence=self.detectionCon,
#                                     min_tracking_confidence=self.trackCon)

#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)

#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
#         return img

#     def getPosition(self, img):
#         self.lmList = []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 self.lmList.append([id, cx, cy])
#         return self.lmList
    

#     # show the fps of the video capture

#     def showFps(self, img):
#         cTime = time.time()
#         fps = 1 / (cTime - self.pTime)
#         self.pTime = cTime
#         cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


#     # checking if squat is being performed or not
#     def isSquat(self):
#         if self.lmList:
#             hip = self.lmList[24]  # Right hip
#             knee = self.lmList[26]  # Right knee
#             ankle = self.lmList[28]  # Right ankle

#             if (abs(hip[2] - knee[2]) <= 30) and (abs(knee[2] - ankle[2]) <= 30):
#                 return True
#         return False
    

#     # checking for left bicep curl

#     def isLeftBicepCurl(self):

#         if self.lmList:
#             left_wrist = self.lmList[15]  # Left wrist
#             left_shoulder = self.lmList[11]  # Left shoulder

#             if abs(left_wrist[2] - left_shoulder[2]) <= 50:
#                 return True
#         return False



# def main():
#     detector = poseDetector()
#     cap = cv2.VideoCapture(0)
#     squat_count = 0
#     left_bicep_curl_count = 0
#     squat_flag = False
#     left_bicep_curl_flag = False

#     while True:
#         success, img = cap.read()
#         if not success:
#             print("Failed to capture image")
#             break
#         img = detector.findPose(img)
#         lmList = detector.getPosition(img)
        
#         if detector.isSquat():
#             if not squat_flag:
#                 squat_count += 1
#                 squat_flag = True

#         else:
#              if squat_flag:
#                 squat_flag = False

#         if detector.isLeftBicepCurl():
#             if not squat_flag:
#                 left_bicep_curl_count += 1
#                 left_bicep_curl_flag = True

#         else:
#              if left_bicep_curl_flag:
#                 left_bicep_curl_flag = False

    

#         # detector.showFps(img)
#         cv2.putText(img, f'Squats: {squat_count}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
#         cv2.putText(img, f'Left Bicep Curls: {left_bicep_curl_count}', (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (10, 255, 0),2)
#         cv2.imshow("Image", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import mediapipe as mp
import time
import threading

class poseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = 0
        self.squat_count = 0
        self.left_bicep_curl_count = 0
        self.squat_flag = False
        self.left_bicep_curl_flag = False

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                    smooth_landmarks=self.smooth,
                                    min_detection_confidence=self.detectionCon,
                                    min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def showFps(self, img):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    def isSquat(self):
        if self.lmList:
            hip = self.lmList[24]  # Right hip
            knee = self.lmList[26]  # Right knee
            ankle = self.lmList[28]  # Right ankle

            if (abs(hip[2] - knee[2]) <= 30) and (abs(knee[2] - ankle[2]) <= 30):
                if not self.squat_flag:
                    self.squat_count += 1
                    self.squat_flag = True
            else:
                if self.squat_flag:
                    self.squat_flag = False

    def isLeftBicepCurl(self):
        if self.lmList:
            left_wrist = self.lmList[15]  # Left wrist
            left_shoulder = self.lmList[11]  # Left shoulder

            if abs(left_wrist[2] - left_shoulder[2]) <= 50:
                if not self.left_bicep_curl_flag:
                    self.left_bicep_curl_count += 1
                    self.left_bicep_curl_flag = True
            else:
                if self.left_bicep_curl_flag:
                    self.left_bicep_curl_flag = False

    def startSquatChecker(self):
        while True:
            self.isSquat()
            time.sleep(4)

    def startLeftBicepCurlChecker(self):
        while True:
            self.isLeftBicepCurl()
            time.sleep(4)

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(0)

    # Start threads for squat and bicep curl checkers
    squat_thread = threading.Thread(target=detector.startSquatChecker)
    bicep_curl_thread = threading.Thread(target=detector.startLeftBicepCurlChecker)
    
    squat_thread.start()
    bicep_curl_thread.start()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
        img = detector.findPose(img)
        lmList = detector.getPosition(img)

        detector.showFps(img)
        cv2.putText(img, f'Squats: {detector.squat_count}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f'Left Bicep Curls: {detector.left_bicep_curl_count}', (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (10, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
