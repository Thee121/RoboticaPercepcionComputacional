import cv2
import joblib
import numpy as np
from skimage.feature import hog

# Cargar modelo
model = joblib.load("symbol_knn_hog.pkl")
knn = model['knn']
HOG_PARAMS = model['hog_params']
IMAGE_SIZE = model['image_size']
labels_inv = model['labels_inv']

# Parámetros para segmentación rojo
lower_red1 = np.array([0, 100, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 50])
upper_red2 = np.array([180, 255, 255])
kernel = np.ones((5, 5), np.uint8)

# Cargar vídeo
cap = cv2.VideoCapture("Video/video_prueba_t4.avi")
if not cap.isOpened():
    print("No puedo abrir el vídeo")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("resultado_knn_hog.avi",
                      cv2.VideoWriter_fourcc(*'XVID'),
                      fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, lower_red1, upper_red1)
    m2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) >= 300:
            x, y, wc, hc = cv2.boundingRect(cnt)

            # Seguridad para evitar ROI fuera del frame
            x_end = min(x + wc, frame.shape[1])
            y_end = min(y + hc, frame.shape[0])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[y:y_end, x:x_end]

            if roi.size != 0:
                roi_resized = cv2.resize(roi, IMAGE_SIZE)
                feat = hog(roi_resized, **HOG_PARAMS)
                pred = knn.predict([feat])[0]
                label = labels_inv[pred]

                # Dibujar
                cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Detección", frame)
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Vídeo resultado guardado en resultado_knn_hog.avi")
