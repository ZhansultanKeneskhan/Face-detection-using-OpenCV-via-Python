import cv2
import dlib
import numpy as np
import math
import time
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Dastan', 'Zhanat', 'Rustam', 'Zhanik'] 

webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define min window size to be recognized as a face
minW = 0.1*webcam.get(3)
minH = 0.1*webcam.get(4)

pTime = 0

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class Object:
    def __init__(self, size=50):
        self.logo_org = cv2.imread('zhanat.jpg')
        self.size = size
        self.logo = cv2.resize(self.logo_org, (size, size))
        img2gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        _, logo_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        self.logo_mask = logo_mask
        self.x = 295
        self.y = 240
        self.score = 0

    def insert_object(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.logo_mask)] = 0
        roi += self.logo

    def update_position(self, x,y):
        return self.x, self.y

        # Check for collision
def collision(rleft, rtop, width, height,   # rectangle definition
              center_x, center_y, radius):  # circle definition
    """ Detect collision between a rectangle and circle. """

    # complete boundbox of the rectangle
    rright, rbottom = rleft + width, rtop + height

    # bounding box of the circle
    cleft, ctop     = center_x-radius, center_y-radius
    cright, cbottom = center_x+radius, center_y+radius

    # trivial reject if bounding boxes do not intersect
    if rright < cleft or rleft > cright or rbottom < ctop or rtop > cbottom:
        return False  # no collision possible

    # check whether any point of rectangle is inside circle's radius
    for x in (rleft, rleft+width):
        for y in (rtop, rtop+height):
            # compare distance between circle's center point and each point of
            # the rectangle with the circle's radius
            if math.hypot(x-center_x, y-center_y) <= radius:
                return True  # collision detected

    # check if center of circle is inside rectangle
    if rleft <= center_x <= rright and rtop <= center_y <= rbottom:
        return True  # overlaid

    return False  # no collision detected


# Let's create the object
obj1 = Object()
obj2 = Object()
while True:
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,f'FPS:{int(fps)}', (280,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.line(frame, (320,0), (320,480), (0,255,0), 1)
    faces = hog_face_detector(gray)
    noses_x = []
    noses_y = []
    
         
    for face in range(0,len(faces)):
        face_landmarks = dlib_facelandmark(gray, faces[face])
        nose_x = face_landmarks.part(30).x
        nose_y = face_landmarks.part(30).y
        noses_x.append(nose_x)
        noses_y.append(nose_y)
        rad = 5
        cv2.circle(frame, (nose_x, nose_y), rad, (0, 255, 255), cv2.FILLED)
        cv2.putText(frame, str(30), (nose_x,nose_y-10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255), 1)
        obj1.insert_object(frame)
        obj2.insert_object(frame)
        if (len(faces) == 2):
            #it appends elements 1-by-1, so we need to wait until noses have two elements
            if (len(noses_x) == 1):
                pass
            else:
                #for fair score: 1 player should be in left part
                if (noses_x[0] < noses_x[1]):
                    check1 = collision(obj1.x, obj1.y, obj1.size, obj1.size, noses_x[0], noses_y[0], rad)
                    check2 = collision(obj2.x, obj2.y, obj2.size, obj2.size, noses_x[1], noses_y[1], rad)
                else:
                    check2 = collision(obj2.x, obj2.y, obj2.size, obj2.size, noses_x[0], noses_y[0], rad)
                    check1 = collision(obj1.x, obj1.y, obj1.size, obj1.size, noses_x[1], noses_y[1], rad)                    
                #print("Player1:", check1)
                #print("Player2:", check2)
                if (check1 == True):
                    obj1.x = np.random.randint(50, 250 - obj1.size - 1)
                    obj1.y = np.random.randint(50, 450 - obj1.size - 1)
                    obj1.update_position(obj1.x, obj1.y)
                    obj1.score += 1
                elif (check2 == True):
                    obj2.x = np.random.randint(370, 520 - obj2.size - 1)
                    obj2.y = np.random.randint(50, 450 - obj2.size - 1)                    
                    obj2.update_position(obj2.x, obj2.y)
                    obj2.score += 1    
    trained_faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        
    for(x,y,w,h) in trained_faces:

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            text1 = "Player1:"+str(obj1.score)
            text2 = "Player2:"+str(obj2.score)
            cv2.putText(frame, text1, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            cv2.putText(frame, text2, (400, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)     

        #for n in range(0, 68):
         #   x = face_landmarks.part(n).x
          #  y = face_landmarks.part(n).y
           # cv2.circle(frame, (x, y), 2, (0, 255, 255), cv2.FILLED)
            #cv2.putText(frame, str(n), (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255), 1) 

    cv2.imshow("Face Landmarks", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'): 
        webcam.release() 
        cv2.destroyAllWindows() 
        break  