import cv2
import math
from threading import Thread
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class VideoGet:

    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()      

    def stop(self):
        self.stopped = True 

class VideoShow:

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Hand Tracking Output", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True   

def threadshow():

    video_getter = VideoGet().start()
    video_shower = VideoShow(video_getter.frame).start()  

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame

        video_shower.frame = frame

class threadVideo:

    def __init__(self):
        self.video_getter = VideoGet().start()
        self.video_shower = VideoShow(self.video_getter.frame).start()
        self.dist = 1.0

    def start(self):
        Thread(target=self.show, args=()).start()
        return self   

    def show(self):
        with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
            while True:
                if self.video_getter.stopped or self.video_shower.stopped:
                    self.video_shower.stop()
                    self.video_getter.stop()
                    break

                image = self.video_getter.frame
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                dist= 1.0
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        thumb = hand_landmarks.landmark[4]
                        middle = hand_landmarks.landmark[10]
                        dist = math.sqrt((thumb.x - middle.x)**2 + (thumb.y - middle.y)**2)
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                self.video_shower.frame = image
                self.dist = dist
if __name__ == '__main__':   
    threadVideo().start()
