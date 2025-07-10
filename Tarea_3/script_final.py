import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# ----------------------------
# 1. Cargar muestras y entrenar clasificador
# ----------------------------

# Cargar tu CSV real
df = pd.read_csv("Resultado/samples.csv")
X = df[['R', 'G', 'B']].values
y = df['Label'].values

clf = SVC()
clf.fit(X, y)

print("¡Clasificador entrenado!")

# ----------------------------
# 2. Preparar vídeo de entrada
# ----------------------------

REAL_DIAMETER_CM = 7.0     # cámbialo si sabes el diámetro real
FOCAL_LENGTH_PX = 800      # cámbialo si sabes la focal real

VIDEO_PATH = "Videos_Bola/Tenis2.avi"
OUTPUT_PATH = "output_video.avi"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening video.")
    exit()

# Obtener info del vídeo original
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Definir codec y VideoWriter para guardar vídeo
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # o 'mp4v' para mp4
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Parámetro nuevo → área mínima
MIN_AREA = 80   # ajusta según el tamaño de tu pelota en píxeles

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Clasificar cada pixel
    pixels = frame.reshape((-1, 3))
    predictions = clf.predict(pixels)
    mask = predictions.reshape(frame.shape[:2]).astype(np.uint8) * 255

    # Suavizar máscara
    mask_blur = cv2.medianBlur(mask, 7)

    # Buscar contornos
    contours, _ = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    diameter_px = None
    distance_cm = None
    detectado = False

    if contours:
        # Escoger el mayor contorno
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        print("Área del contorno detectado:", area)


        if area > MIN_AREA:
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)

            if radius > 5:
                center = (int(x), int(y))
                radius = int(radius)
                diameter_px = 2 * radius

                # Calcular distancia real si tienes datos reales
                distance_cm = (REAL_DIAMETER_CM * FOCAL_LENGTH_PX) / diameter_px

                # Dibujar círculo en el frame
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

                detectado = True

    # Dibujar texto en el frame
    if detectado:
        text1 = f"Diameter: {diameter_px}px"
        text2 = f"Distance: {distance_cm:.1f} cm"
        color = (0, 255, 255)
    else:
        text1 = "NO DETECTA"
        text2 = ""
        color = (0, 0, 255)

    cv2.putText(
        frame,
        text1,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    if text2:
        cv2.putText(
            frame,
            text2,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    # Escribir el frame en el vídeo de salida
    out.write(frame)

    # (opcional) mostrar mientras se procesa
    cv2.imshow("Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"¡Vídeo guardado en {OUTPUT_PATH}!")
