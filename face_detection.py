import cv2

""" 
    https://github.com/Itseez/opencv/blob/master/data/haarcascades 
    Haarcascade xml files used in this project
"""
class FaceDetection(object):
    def __init__(self):        
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def crop_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 1,minSize=(150, 150))      
        if len(faces) == 1:
            (x,y,w,h) = faces[0] 
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) != 2:
                return None
            return roi_color        
        return None