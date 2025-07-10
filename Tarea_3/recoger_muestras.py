import cv2
import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path.cwd()
media_path = base_dir / 'Media'

VIDEO_PATH = media_path / "Videos_Bola/Tenis2.avi"
OUTPUT_CSV = "Resultado/samples.csv"

samples = []
labels = []

# Clase a etiquetar: 1 = pelota, 0 = fondo
current_label = 1

def mouse_callback(event, x, y, flags, param):
    global samples, labels, frame, current_label
    if event == cv2.EVENT_LBUTTONDOWN:
        # Color BGR
        b, g, r = frame[y, x]
        print(f"Clicked pixel at ({x},{y}) → RGB = {r},{g},{b} Label={current_label}")
        samples.append([b, g, r])
        labels.append(current_label)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening video.")
    exit()

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar texto de ayuda
    text = f"Label: {current_label} (Press '1'=pelota, '0'=fondo, 's'=save, 'q'=quit)"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('1'):
        current_label = 1
        print("Etiqueta actual: Pelota (1)")
    elif key == ord('0'):
        current_label = 0
        print("Etiqueta actual: Fondo (0)")
    elif key == ord('s'):
        # Guardar muestras
        df = pd.DataFrame(samples, columns=['R', 'G', 'B'])
        df['Label'] = labels
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Samples saved to {OUTPUT_CSV}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
