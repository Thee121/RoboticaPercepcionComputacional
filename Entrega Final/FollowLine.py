import cv2
from pyrobot.tools.followLineTools import findLineDeviation
import cv2
import joblib
import numpy as np
from skimage.feature import hog

# Parámetros para segmentación rojo
lower_red1 = np.array([0, 100, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 50])
upper_red2 = np.array([180, 255, 255])
kernel = np.ones((5, 5), np.uint8)

class FollowLine:
    def __init__(self, robot):
        self.robot = robot
        self.label_printed = False
        self.no_symbol_count = 0

    def get_image(self):
        image = self.robot.getImage()
        cv2.imshow("Stage Camera Image", image)
        cv2.waitKey(1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, error = findLineDeviation(gray)
        self.error_line = error
        return found

    def follow_line(self):
        error = self.error_line
        correction = 1 - abs(error)
        self.detect_image()
        
        if(error>0.1): #Linea está a derecha
            self.robot.move(0.5 * correction, -error * 4)
            return
        elif(error<0.1): #Linea está izquierda
            self.robot.move(0.5 * correction, -error * 4)
            return
        else: #Linea en el centro
            self.robot.move(0.5, 0.0)
            return
    
    def detect_image(self):
        hsv = cv2.cvtColor(self.robot.getImage(), cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, lower_red1, upper_red1)
        m2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        repeat = 0
        
        if cnts:
            self.no_symbol_count = 0

            if not self.label_printed:
                cnt = max(cnts, key=cv2.contourArea)

                if cv2.contourArea(cnt) >= 300:
                    x, y, wc, hc = cv2.boundingRect(cnt)
                    frame = self.robot.getImage()

                    # Seguridad para evitar ROI fuera del frame
                    x_end = min(x + wc, frame.shape[1])
                    y_end = min(y + hc, frame.shape[0])
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi = gray[y:y_end, x:x_end]
                    
                    if roi.size != 0:

                        # Cargar modelo todas las figuras
                        model = joblib.load("symbol_knn_hog_Todos.pkl")
                        knn = model['knn']

                        HOG_PARAMS = model['hog_params']

                        IMAGE_SIZE = model['image_size']
                        labels_dict = {'Man':0, 'Woman':1, 'Telephone':2, 'Stairs':3, 'Arrow':4}
                        labels_inv = {v: k for k, v in labels_dict.items()}
                        roi_resized = cv2.resize(roi, IMAGE_SIZE)
                        feat = hog(roi_resized, **HOG_PARAMS)
                        pred = knn.predict([feat])[0]
                        label = labels_inv[pred]

                        if(label == "Arrow"):
                            self.cnt = max(cnts, key = cv2.contourArea)
                            self.calculate_direction(cnt)

                        else:
                            print(label)
                            
                        self.label_printed = True

        else:
            self.no_symbol_count += 1

            if self.no_symbol_count >= 2:
                self.label_printed = False
        return



    def calculate_direction(self, cnts):
        IMAGE_SIZE = (64, 64)
        
        # Cargar modelo flechas
        model = joblib.load("symbol_knn_hog_flechas.pkl")
        knn = model['knn']
        HOG_PARAMS = model['hog_params']
        IMAGE_SIZE = model['image_size']
        labels_inv = model['labels_inv']

        knn_model = model['knn']
        x, y, w, h = cv2.boundingRect(cnts)
        gray = cv2.cvtColor(self.robot.getImage(), cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, IMAGE_SIZE)
        feat = hog(roi, **HOG_PARAMS)
        pred = knn_model.predict([feat])[0]
        
        direccion = labels_inv[pred]
        
        if direccion:
            print(f"Flecha detectada, siguiendo camino")
            if direccion == "Right":
                self.robot.move(0.0, -2.0)
            elif direccion == "Forward":
                self.robot.move(2.0, 0.0)
            elif direccion == "Left":
                self.robot.move(0.0, 2.0)
